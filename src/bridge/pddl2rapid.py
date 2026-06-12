"""
pddl_to_rapid_bridge.py
=======================
Translates a PDDL planner file into ABB RAPID motion commands and streams them to the robot controller
over a TCP/IP socket.


Usage
-----
  python pddl_to_rapid_bridge.py \
      --planner  tool_usage_planner.pddl \
      --config   pose_config.yaml \
      --host     192.168.125.1 \
      --port     5000 \
      [--dry-run]
"""

import argparse
import logging
import re
import socket
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("pddl_rapid_bridge")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def to_rapid(self) -> str:
        """Return a RAPID quaternion literal [qw, qx, qy, qz]."""
        return f"[{self.w:.6f},{self.x:.6f},{self.y:.6f},{self.z:.6f}]"


@dataclass
class Position:
    """3-D Cartesian position in millimetres (robot base frame)."""
    x: float
    y: float
    z: float

    def to_rapid(self) -> str:
        """Return a RAPID position literal [x, y, z]."""
        return f"[{self.x:.3f},{self.y:.3f},{self.z:.3f}]"


@dataclass
class RobotTarget:
    """
    Full ABB robtarget: position + orientation + config + external axes.

    RAPID:
        [[px,py,pz],[qw,qx,qy,qz],[cf1,cf4,cf6,cfx],[eax1,eax2,eax3,eax4,eax5,eax6]]
    """
    position: Position
    orientation: Quaternion
    # Robot configuration vector (axis quadrant flags); defaults are valid for most poses
    config: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    # External axes (not used for GoFA without external axes)
    ext_axes: list[float] = field(default_factory=lambda: [9e9, 9e9, 9e9, 9e9, 9e9, 9e9])

    def to_rapid(self) -> str:
        cfg = ",".join(str(c) for c in self.config)
        ext = ",".join(f"{e:.1f}" for e in self.ext_axes)
        return (
            f"[{self.position.to_rapid()},"
            f"{self.orientation.to_rapid()},"
            f"[{cfg}],"
            f"[{ext}]]"
        )


@dataclass
class PDDLAction:
    """One grounded PDDL action as parsed from the planner file."""
    name: str
    params: list[str]

    def __repr__(self) -> str:
        return f"({self.name} {' '.join(self.params)})"


# ---------------------------------------------------------------------------
# PDDL Planner Parser
# ---------------------------------------------------------------------------

class PDDLPlanParser:
    """
    Parses a PDDL planner output file.

    """

    # Match a single grounded action: ( action params... )
    _ACTION_RE = re.compile(
        r"^\(\s*([a-zA-Z_][a-zA-Z0-9_\-]*)"  # action name
        r"((?:\s+[a-zA-Z_][a-zA-Z0-9_\-]*)*)"  # zero or more params
        r"\s*\)$"
    )

    def parse(self, planner_path: Path) -> list[PDDLAction]:
        """Return the ordered list of grounded actions from *planner_path*."""
        actions: list[PDDLAction] = []
        text = planner_path.read_text(encoding="utf-8")

        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line or line.startswith(";"):
                continue  # skip comments and blank lines

            # strip inline comments
            line = line.split(";")[0].strip()
            if not line:
                continue

            m = self._ACTION_RE.match(line)
            if m:
                name = m.group(1).lower()
                params = m.group(2).split() if m.group(2).strip() else []
                actions.append(PDDLAction(name=name, params=params))
                logger.debug("Parsed action [line %d]: %s %s", lineno, name, params)
            else:
                logger.warning("Unrecognised line %d (skipped): %r", lineno, raw)

        logger.info("Parsed %d actions from %s", len(actions), planner_path.name)
        return actions


# ---------------------------------------------------------------------------
# Pose Registry
# ---------------------------------------------------------------------------

class PoseRegistry:
    """
    Loads the YAML pose config and resolves symbolic location/orientation
    names to RobotTarget / Quaternion instances.
    """

    def __init__(self, config_path: Path) -> None:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._locations: dict[str, dict] = data.get("locations", {})
        self._orientations: dict[str, dict] = data.get("orientations", {})
        logger.info(
            "PoseRegistry loaded: %d locations, %d orientations",
            len(self._locations),
            len(self._orientations),
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_target(self, location_name: str) -> RobotTarget:
        """Return the full RobotTarget for *location_name*."""
        if location_name not in self._locations:
            raise KeyError(
                f"Location '{location_name}' not found in pose config. "
                f"Available: {list(self._locations.keys())}"
            )
        cfg = self._locations[location_name]
        pos = cfg["position"]
        quat = cfg["quaternion"]
        return RobotTarget(
            position=Position(*pos),
            orientation=Quaternion(*quat),
        )

    def get_orientation(self, orientation_name: str) -> Quaternion:
        """Return the Quaternion for *orientation_name*."""
        if orientation_name not in self._orientations:
            raise KeyError(
                f"Orientation '{orientation_name}' not found in pose config. "
                f"Available: {list(self._orientations.keys())}"
            )
        quat = self._orientations[orientation_name]["quaternion"]
        return Quaternion(*quat)

    def get_speed(self, location_name: str) -> str:
        """Return the RAPID speed data identifier for *location_name*."""
        return self._locations.get(location_name, {}).get("speed_data", "v100")

    def get_zone(self, location_name: str) -> str:
        """Return the RAPID zone identifier for *location_name*."""
        return self._locations.get(location_name, {}).get("zone_data", "z10")

    def get_motion_type(self, location_name: str) -> str:
        """Return 'MoveJ' or 'MoveL' for *location_name*."""
        return self._locations.get(location_name, {}).get("motion_type", "MoveJ")

    def patch_orientation(
        self, target: RobotTarget, orientation_name: str
    ) -> RobotTarget:
        """
        Return a copy of *target* with the orientation replaced by the
        one registered under *orientation_name*.
        """
        new_quat = self.get_orientation(orientation_name)
        return RobotTarget(
            position=target.position,
            orientation=new_quat,
            config=target.config,
            ext_axes=target.ext_axes,
        )


# ---------------------------------------------------------------------------
# RAPID Command Builder
# ---------------------------------------------------------------------------

class RAPIDCommandBuilder:
    """
    Builds RAPID instruction strings.

    All methods return a list of strings, one per logical instruction,
    because a single PDDL action may expand to multiple RAPID lines
    (e.g. grasp → MoveJ + EGP_CLOSE).

    RAPID instruction reference (Omnicore controller):
        MoveJ  <robtarget>, <speeddata>, <zonedata>, tool0;
        MoveL  <robtarget>, <speeddata>, <zonedata>, tool0;
        EGP_OPEN   – open the Parallel Electric Gripper
        EGP_CLOSE  – close the Parallel Electric Gripper
    """

    # Default tool name used in all motion instructions
    TOOL = "gripper"
    # Default work object
    WOBJ = "wobj0"

    def move_j(
        self,
        target: RobotTarget,
        speed: str = "v100",
        zone: str = "z10",
    ) -> str:
        """Joint-space motion to target."""
        return (
            f"MoveJ {target.to_rapid()},{speed},{zone},{self.TOOL}\\WObj:={self.WOBJ};"
        )

    def move_l(
        self,
        target: RobotTarget,
        speed: str = "v50",
        zone: str = "fine",
    ) -> str:
        """Linear Cartesian motion to target."""
        return (
            f"MoveL {target.to_rapid()},{speed},{zone},{self.TOOL}\\WObj:={self.WOBJ};"
        )

    def egp_open(self) -> str:
        """Open the Electric Parallel Gripper."""
        return "EGP_OPEN;"

    def egp_close(self) -> str:
        """Close the Electric Parallel Gripper."""
        return "EGP_CLOSE;"

    def wait_io(self, signal: str, value: int = 1, timeout: float = 5.0) -> str:
        """
        Wait for a digital input to reach *value* (used for grasp verification).
        Translates to RAPID WaitDI with a timeout.
        """
        return f"WaitDI {signal},{value}\\MaxTime:={timeout:.1f};"

    def comment(self, text: str) -> str:
        """Insert a RAPID comment line (sent as metadata, not executed)."""
        return f"! {text}"


# ---------------------------------------------------------------------------
# PDDL → RAPID Translator
# ---------------------------------------------------------------------------

class PDDLToRAPIDTranslator:
    """
    Translates a list of PDDLAction objects into sequences of RAPID
    instruction strings using the PoseRegistry and RAPIDCommandBuilder.

    Each PDDL action is handled by a dedicated private method.
    Unknown actions raise a TranslationError.
    """

    # Digital input signal name for grip confirmation (controller-side)
    GRIP_CONFIRM_SIGNAL = "di_grip_ok"

    def __init__(self, pose_registry: PoseRegistry) -> None:
        self._reg = pose_registry
        self._builder = RAPIDCommandBuilder()
        # Track the last known orientation descriptor for orient actions
        self._current_orientation: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def translate(self, actions: list[PDDLAction]) -> list[tuple[PDDLAction, list[str]]]:
        """
        Translate *actions* and return a list of (action, [rapid_instructions])
        pairs preserving order.
        """
        result: list[tuple[PDDLAction, list[str]]] = []
        for action in actions:
            instructions = self._dispatch(action)
            result.append((action, instructions))
        return result

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def _dispatch(self, action: PDDLAction) -> list[str]:
        """Route *action* to the appropriate handler method."""
        handler = getattr(self, f"_handle_{action.name}", None)
        if handler is None:
            raise TranslationError(
                f"No RAPID translation defined for PDDL action '{action.name}'. "
                f"Add a _handle_{action.name}() method to PDDLToRAPIDTranslator."
            )
        return handler(action)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_move(self, action: PDDLAction) -> list[str]:
        """
        (move ?robot ?from ?to)
        → MoveJ to the destination location.

        The motion type (MoveJ / MoveL) is read from the pose config
        of the destination location.
        """
        # params: [robot, from_loc, to_loc]
        if len(action.params) < 3:
            raise TranslationError(f"'move' expects 3 params, got: {action.params}")

        to_loc = action.params[2]
        target = self._reg.get_target(to_loc)
        speed  = self._reg.get_speed(to_loc)
        zone   = self._reg.get_zone(to_loc)
        motion = self._reg.get_motion_type(to_loc)

        cmds = [self._builder.comment(f"move → {to_loc}")]
        if motion == "MoveL":
            cmds.append(self._builder.move_l(target, speed, zone))
        else:
            cmds.append(self._builder.move_j(target, speed, zone))
        return cmds

    def _handle_grasp(self, action: PDDLAction) -> list[str]:
        """
        (grasp ?robot ?tool ?grasp_point ?location)
        → MoveJ to the tool location (pre-grasp approach handled by planner)
          + EGP_CLOSE
        """
        # params: [robot, tool, grasp_point, location]
        if len(action.params) < 4:
            raise TranslationError(f"'grasp' expects 4 params, got: {action.params}")

        loc   = action.params[3]
        target = self._reg.get_target(loc)
        speed  = self._reg.get_speed(loc)
        zone   = self._reg.get_zone(loc)

        tool_name  = action.params[1]
        grasp_name = action.params[2]

        return [
            self._builder.comment(f"grasp {tool_name} at {grasp_name}"),
            self._builder.move_j(target, speed, zone),
            self._builder.egp_close(),
        ]

    def _handle_verify_grasp(self, action: PDDLAction) -> list[str]:
        """
        (verify_grasp ?robot ?tool ?grasp_point)
        → WaitDI on grip-confirmation digital input.
          No physical motion is required; this is a controller-side check.
        """
        tool_name = action.params[1] if len(action.params) > 1 else "tool"
        return [
            self._builder.comment(f"verify_grasp – waiting for grip confirmation on {tool_name}"),
            self._builder.wait_io(self.GRIP_CONFIRM_SIGNAL, value=1, timeout=5.0),
        ]

    def _handle_orient_vertical(self, action: PDDLAction) -> list[str]:
        """
        (orient_vertical ?robot ?tool ?orientation ?location)
        → MoveJ to the same Cartesian position with the vertical quaternion applied.

        The current location is read from the last known position (param[3]).
        """
        return self._apply_orientation(action, orientation_name="vertical_orient")

    def _handle_orient_hammer_strike(self, action: PDDLAction) -> list[str]:
        """
        (orient_hammer_strike ?robot ?tool ?orientation ?location)
        → MoveJ to the same Cartesian position with the hammer-strike quaternion applied.
        """
        return self._apply_orientation(action, orientation_name="hammer_strike_orient")

    def _handle_approach_target(self, action: PDDLAction) -> list[str]:
        """
        (approach_target ?robot ?tool ?target ?above_target)
        → MoveJ to the above-target waypoint (already in the pose config).
          The robot should already be near this location after the prior move;
          this step enforces the correct approach axis.
        """
        # params: [robot, tool, target_loc, above_target_loc]
        if len(action.params) < 4:
            raise TranslationError(
                f"'approach_target' expects 4 params, got: {action.params}"
            )

        above_loc = action.params[3]
        target    = self._reg.get_target(above_loc)
        speed     = self._reg.get_speed(above_loc)
        zone      = self._reg.get_zone(above_loc)

        return [
            self._builder.comment(f"approach_target – descend to pre-contact waypoint {above_loc}"),
            self._builder.move_j(target, speed, zone),
        ]

    def _handle_place(self, action: PDDLAction) -> list[str]:
        """
        (place ?robot ?tool ?tip ?target ?above_target)
        → MoveL (linear descent) to the contact target + EGP_OPEN to release.
        """
        # params: [robot, tool, tip, target_loc, above_target_loc]
        if len(action.params) < 5:
            raise TranslationError(f"'place' expects 5 params, got: {action.params}")

        target_loc = action.params[3]
        target     = self._reg.get_target(target_loc)
        speed      = self._reg.get_speed(target_loc)
        zone       = self._reg.get_zone(target_loc)
        tool_name  = action.params[1]
        tip_name   = action.params[2]

        return [
            self._builder.comment(
                f"place {tool_name} tip ({tip_name}) onto {target_loc}"
            ),
            self._builder.move_l(target, speed, zone),   # linear descent
            self._builder.egp_open(),                     # release tool
        ]

    def _handle_release(self, action: PDDLAction) -> list[str]:
        """
        (release ?robot ?tool ?location)
        → EGP_OPEN (abort grasp or intermediate re-grasp).
        """
        tool_name = action.params[1] if len(action.params) > 1 else "tool"
        return [
            self._builder.comment(f"release {tool_name}"),
            self._builder.egp_open(),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_orientation(
        self, action: PDDLAction, orientation_name: str
    ) -> list[str]:
        """
        Shared logic for orient_vertical / orient_hammer_strike.
        Retrieves the current location from the action params, patches
        its orientation, and emits a MoveJ to the same position with
        the new quaternion.
        """
        # params: [robot, tool, orientation_descriptor, location]
        if len(action.params) < 4:
            raise TranslationError(
                f"'{action.name}' expects 4 params, got: {action.params}"
            )

        loc    = action.params[3]
        target = self._reg.get_target(loc)
        patched = self._reg.patch_orientation(target, orientation_name)
        speed  = self._reg.get_speed(loc)
        # Use 'fine' zone for orientation changes to ensure exact pose is reached
        zone   = "fine"

        self._current_orientation = orientation_name

        return [
            self._builder.comment(
                f"{action.name} – reorient at {loc} using '{orientation_name}'"
            ),
            self._builder.move_j(patched, speed, zone),
        ]


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class TranslationError(RuntimeError):
    """Raised when a PDDL action cannot be translated to RAPID."""


# ---------------------------------------------------------------------------
# TCP/IP Socket Client
# ---------------------------------------------------------------------------

class RobotSocketClient:
    """
    Thin TCP/IP wrapper that streams RAPID command strings to the ABB
    controller and waits for an ACK after each instruction.

    Protocol
    --------
    - Each command is sent as UTF-8 text followed by '\\n'.
    - The controller responds with 'ACK\\n' on success or 'ERR <msg>\\n' on failure.
    - On ERR the client raises RobotCommunicationError.
    - recv_timeout: seconds to wait for ACK before raising TimeoutError.
    """

    ENCODING = "utf-8"
    ACK      = "ACK"
    ERR_PREFIX = "ERR"
    BUFFER_SIZE = 1024

    def __init__(
        self,
        host: str,
        port: int,
        recv_timeout: float = 10.0,
        connect_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.recv_timeout = recv_timeout
        self.connect_timeout = connect_timeout
        self._sock: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "RobotSocketClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open TCP connection to the robot controller."""
        logger.info("Connecting to robot at %s:%d …", self.host, self.port)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        sock.connect((self.host, self.port))
        sock.settimeout(self.recv_timeout)
        self._sock = sock
        logger.info("Connected.")

    def close(self) -> None:
        """Close the TCP connection."""
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
            logger.info("Socket closed.")

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    def send_command(self, command: str) -> None:
        """
        Send a single RAPID instruction and block until ACK is received.

        Parameters
        ----------
        command : str
            A RAPID instruction string (without trailing newline).

        Raises
        ------
        RobotCommunicationError
            If the controller replies with ERR or the reply is unexpected.
        TimeoutError
            If no reply arrives within recv_timeout seconds.
        RuntimeError
            If the socket is not connected.
        """
        if self._sock is None:
            raise RuntimeError("Socket is not connected. Call connect() first.")

        payload = (command + "\n").encode(self.ENCODING)
        logger.debug("TX: %s", command)

        self._sock.sendall(payload)
        response = self._receive_line()

        if response.startswith(self.ACK):
            logger.debug("RX: ACK")
        elif response.startswith(self.ERR_PREFIX):
            raise RobotCommunicationError(
                f"Controller returned error for command '{command}': {response}"
            )
        else:
            raise RobotCommunicationError(
                f"Unexpected response from controller: '{response}'"
            )

    def _receive_line(self) -> str:
        """Read bytes until '\\n' and return the decoded stripped line."""
        buf = b""
        while True:
            chunk = self._sock.recv(self.BUFFER_SIZE)
            if not chunk:
                raise RobotCommunicationError(
                    "Connection closed by controller while waiting for ACK."
                )
            buf += chunk
            if b"\n" in buf:
                line, _ = buf.split(b"\n", maxsplit=1)
                return line.decode(self.ENCODING).strip()


class RobotCommunicationError(RuntimeError):
    """Raised on unexpected or error responses from the robot controller."""


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class PDDLToRAPIDPipeline:
    """
    End-to-end pipeline:
      1. Parse the PDDL planner file.
      2. Translate each action to RAPID instructions.
      3. Stream instructions to the robot over TCP/IP (or print in dry-run mode).
    """

    def __init__(
        self,
        planner_path: Path,
        config_path: Path,
        host: str,
        port: int,
        dry_run: bool = False,
        inter_action_delay: float = 0.1,
    ) -> None:
        self.planner_path = planner_path
        self.config_path  = config_path
        self.host = host
        self.port = port
        self.dry_run = dry_run
        self.inter_action_delay = inter_action_delay

    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full pipeline."""
        logger.info("=== PDDL → RAPID Bridge ===")
        logger.info("Planner : %s", self.planner_path)
        logger.info("Config  : %s", self.config_path)
        logger.info("Target  : %s:%d", self.host, self.port)
        logger.info("Dry-run : %s", self.dry_run)

        # Step 1: Parse PDDL plan
        parser   = PDDLPlanParser()
        actions  = parser.parse(self.planner_path)

        # Step 2: Translate to RAPID
        registry   = PoseRegistry(self.config_path)
        translator = PDDLToRAPIDTranslator(registry)
        plan       = translator.translate(actions)

        # Flatten for logging / summary
        total_instructions = sum(len(cmds) for _, cmds in plan)
        logger.info(
            "Translation complete: %d actions → %d RAPID instructions",
            len(plan),
            total_instructions,
        )

        # Step 3: Stream to controller (or print in dry-run mode)
        if self.dry_run:
            self._dry_run_print(plan)
        else:
            self._stream_to_robot(plan)

        logger.info("=== Pipeline complete ===")

    # ------------------------------------------------------------------

    def _dry_run_print(
        self, plan: list[tuple[PDDLAction, list[str]]]
    ) -> None:
        """Print all RAPID instructions to stdout without connecting."""
        print("\n" + "=" * 60)
        print("DRY-RUN: RAPID instructions (not sent to robot)")
        print("=" * 60)
        for action, cmds in plan:
            print(f"\n; --- {action} ---")
            for cmd in cmds:
                print(f"  {cmd}")
        print("\n" + "=" * 60 + "\n")

    def _stream_to_robot(
        self, plan: list[tuple[PDDLAction, list[str]]]
    ) -> None:
        """Connect to the robot and stream all instructions with ACK checking."""
        with RobotSocketClient(self.host, self.port) as client:
            for action, cmds in plan:
                logger.info("Executing action: %s", action)
                for cmd in cmds:
                    if cmd.startswith("!"):
                        # Send comments as informational messages but still
                        # wait for ACK so the controller can log them
                        client.send_command(cmd)
                    else:
                        client.send_command(cmd)
                    time.sleep(self.inter_action_delay)
                logger.info("Action complete: %s", action)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Translate a PDDL plan into ABB RAPID commands and stream "
                    "them to the robot controller over TCP/IP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--planner", required=True, type=Path,
        help="Path to the PDDL planner output file (e.g. tool_usage_planner.pddl)",
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="Path to the YAML pose configuration file (e.g. pose_config.yaml)",
    )
    p.add_argument(
        "--host", default="192.168.125.1",
        help="IP address of the ABB robot controller",
    )
    p.add_argument(
        "--port", type=int, default=5000,
        help="TCP port on the robot controller",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print translated RAPID instructions without connecting to the robot",
    )
    p.add_argument(
        "--delay", type=float, default=0.1,
        help="Delay in seconds between consecutive instruction sends",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG-level logging",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = PDDLToRAPIDPipeline(
        planner_path=args.planner,
        config_path=args.config,
        host=args.host,
        port=args.port,
        dry_run=args.dry_run,
        inter_action_delay=args.delay,
    )
    pipeline.run()


if __name__ == "__main__":
    main()