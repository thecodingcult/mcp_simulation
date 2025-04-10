import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button  # For Reset Button
import numpy as np
import math
import time  # For potential frame timing debugging

# --- Configuration ---

# Theme & Colors (Refined Dark Mode)
FIG_BG_COLOR = '#1A1A1A'  # Slightly darker background
COMPONENT_BG_COLOR = '#2C2C2C'  # Component background
BORDER_COLOR = '#4F4F4F'  # Default border
ACTIVE_BORDER_COLOR = '#00BFFF'  # Deep Sky Blue - high visibility
GLOW_COLOR = '#00BFFF'  # Match active border
TEXT_COLOR = '#F0F0F0'  # Slightly brighter text
TEXT_LABEL_COLOR = '#B0B0B0'  # Lighter label text
PATH_COLOR = '#555555'  # Default path
ACTIVE_PATH_COLOR = '#00BFFF'  # Match active border
INACTIVE_PATH_ALPHA = 0.5  # Make inactive paths fainter
PACKET_COLORS = {  # Consistent palette, consider accessibility (e.g., colorblindness)
    "default": '#FFA500',  # Orange
    "query": '#FF6347',  # Tomato Red
    "memory_query": '#87CEEB',  # Sky Blue
    "memory_context": '#98FB98',  # Pale Green
    "tool_call": '#FFD700',  # Gold
    "tool_result": '#DAA520',  # Goldenrod
    "context_data": '#DDA0DD',  # Plum
    "final_prompt": '#FA8072',  # Salmon
    "response_stream": '#48D1CC',  # Medium Turquoise
    "update_summary": '#B0C4DE',  # Light Steel Blue
    "update_fact": '#C0C0C0',  # Silver
}
NARRATION_BG = '#202020'
BUTTON_BG = '#3D3D3D'
BUTTON_HOVER_BG = '#5A5A5A'
BUTTON_TEXT_COLOR = '#E0E0E0'
INFO_BG_COLOR = '#252B30'  # Dark slate blue/gray for info box
INFO_TEXT_COLOR = '#E0E0E0'

# Layout & Style
FIG_SIZE = (17, 11)  # Slightly larger figure
COMPONENT_PADDING = 0.015
LABEL_HEIGHT = 0.03
CORNER_RADIUS = 0.01  # Smoother corners (used for components, not buttons now)
BORDER_WIDTH = 1.2
ACTIVE_BORDER_WIDTH = 2.8
PACKET_SIZE = 0.0085  # Slightly larger packets
PATH_LINEWIDTH = 1.2  # Default path thickness
ACTIVE_PATH_LINEWIDTH = 2.8  # Active path thickness
INACTIVE_COMPONENT_ALPHA = 0.85  # Make inactive components slightly transparent
INFO_PANEL_WIDTH = 0.24  # Slightly wider info panel
CONTROL_PANEL_HEIGHT = 0.05  # Space for Play/Pause/Speed buttons

# Fonts
# Prioritize common cross-platform sans-serif fonts, end with generic 'sans-serif'
FONT_FAMILY = ('Helvetica Neue', 'Arial', 'Helvetica', 'Verdana', 'Geneva',
               'sans-serif')
# Reduced font sizes slightly as a precaution against overflow
FONT_SIZE_LABEL = 10.5
FONT_SIZE_NARRATION = 12.0
FONT_SIZE_IO = 9.8  # <<< Reduced
FONT_SIZE_INFO = 9.5  # <<< Reduced
FONT_WEIGHT_LABEL = 'normal'

# Animation Timing & Control
DEFAULT_FRAME_INTERVAL_MS = 35  # Base speed (can be adjusted)
PACKET_DURATION_FACTOR = 0.9  # <<< INCREASED to reduce visual pause between packet arrival and next step activation
EASING_ENABLED = True  # Use cosine easing for smoother packet movement
GLOW_PULSE_SPEED = 0.25  # Speed of the active component glow pulse

# --- Simulation Data ---
USER_QUERY = "What’s on my calendar today?"
MEMORY_QUERY = "calendar events today"
MEMORY_CONTENT_INITIAL = "Stored Context:\n- Flight KL123 booked\n- Reminder: Call Mom\n- Project 'X' deadline soon"
MEMORY_SEARCH_RESULT = "Found Related:\nCalendar usage\npattern (mid-mornings)"  # Already fixed
# *** MODIFIED MEMORY_RETRIEVED FOR BETTER FIT ***
MEMORY_RETRIEVED = "Context: User checks\ncalendar mid-morning."  # Added newline
TOOL_NAME = "CalendarAPI"
TOOL_CALL_DETAILS = f"Call: {TOOL_NAME}"
TOOL_RUNNING_TEXT = f"Running {TOOL_NAME}..."
TOOL_RESULT = "Result:\n- Team Sync @ 10 AM\n- Project Standup @ 2 PM"
ASSEMBLER_INPUT = "Receiving Data:\n- Memory Context\n- Tool Result"
ASSEMBLER_PROCESSING = "Assembling Prompt..."
ASSEMBLED_CONTEXT_LABEL = "Assembled Prompt"
LLM_INPUT_LABEL = "Final Prompt"
LLM_PROCESSING = "LLM Reasoning..."
LLM_OUTPUT_STREAM = [
    " Okay", ",", " checking", " the calendar", ":\n\n", " • Team", " Sync",
    " at 10", ":00 AM", "\n • Project", " Standup", " at 2", ":00 PM."
]
FINAL_RESPONSE = "Okay, checking the calendar:\n\n • Team Sync at 10:00 AM\n • Project Standup at 2:00 PM"
MEMORY_UPDATE_ACTION = "Store Summary"
MEMORY_UPDATE_DETAILS = "Learned: User queried calendar."
MEMORY_UPDATE_CONFIRM = "Memory Updated!"

# --- Component Details (Adding simple markdown-like formatting) ---
COMPONENT_DETAILS = {
    "UserInput":
    "**Role:** User Interface\n**Details:**\n- Starting point of the request.\n- Captures the user's query.",
    "MCPServer":
    "**Role:** MCP Server (Orchestrator)\n**Details:**\n- Receives user input.\n- Manages overall flow.\n- Decides memory/tool usage.\n- Interacts with other components.",
    "MemoryStore":
    "**Role:** Memory Store\n**Details:**\n- Stores/Retrieves long-term context.\n- Provides info based on query.\n- Can be updated with new learnings.",
    "ToolSuite":
    "**Role:** Tool Suite\n**Details:**\n- Collection of external functions/APIs.\n- Examples: Calendar, Web Search, DBs.\n- Called by MCP Server when needed.",
    "CtxAssembler":
    "**Role:** Context Assembler\n**Details:**\n- Combines data before LLM.\n- Injects memory, tool results, query.\n- Creates structured prompt.",
    "LLMEngine":
    "**Role:** LLM Engine\n**Details:**\n- Core Large Language Model.\n- Processes final prompt.\n- Generates text response.",
    "OutputRender":
    "**Role:** Output Renderer\n**Details:**\n- Displays final LLM response.\n- Handles formatting/streaming.",
    "MemoryUpdater":
    "**Role:** Memory Updater\n**Details:**\n- Processes summaries to update Memory.\n- Enables system learning.\n- (Often integrated logic).",
}

# --- Component Coordinates & Setup ---
# Adjust main axes area to account for control panel
main_ax_height = 1.0 - CONTROL_PANEL_HEIGHT - 0.02  # Leave gap for controls
main_ax_bottom = CONTROL_PANEL_HEIGHT + 0.01
main_ax_width = 1.0 - INFO_PANEL_WIDTH - 0.05  # Leave gap for info panel
COMPONENT_SPECS = {  # Positions relative to the main axes area (0 to 1 within its bounds)
    # Y positions adjusted slightly based on reduced height
    "UserInput": {
        "pos": [0.05, 0.76, 0.20, 0.18],
        "label": "1. User Input"
    },
    "OutputRender": {
        "pos": [1.0 - 0.25, 0.76, 0.20, 0.18],
        "label": "7. Output Renderer"
    },  # x=0.75, y=0.76, w=0.2, h=0.18
    "MemoryStore": {
        "pos": [0.05, 0.38, 0.20, 0.30],
        "label": "3. Memory Store"
    },
    "MCPServer": {
        "pos": [0.5 - 0.1, 0.58, 0.20, 0.16],
        "label": "2. MCP Server Core"
    },
    "CtxAssembler": {
        "pos": [0.5 - 0.1, 0.38, 0.20, 0.16],
        "label": "5. Context Assembler"
    },
    "ToolSuite": {
        "pos": [1.0 - 0.25, 0.38, 0.20, 0.30],
        "label": "4. Tool Suite"
    },
    "MemoryUpdater": {
        "pos": [0.05, 0.13, 0.20, 0.16],
        "label": "8. Memory Updater"
    },
    "LLMEngine": {
        "pos": [0.5 - 0.1, 0.13, 0.20, 0.16],
        "label": "6. LLM Engine"
    },  # x=0.4, y=0.13, w=0.2, h=0.16
}

# Calculate derived properties
for name, spec in COMPONENT_SPECS.items():
    x, y, w, h = spec["pos"]
    spec["center"] = (x + w / 2, y + h / 2)
    spec["box_coords"] = (x, y, w, h)
    spec["content_area"] = [
        x + COMPONENT_PADDING, y + COMPONENT_PADDING,
        w - 2 * COMPONENT_PADDING, h - LABEL_HEIGHT - COMPONENT_PADDING * 1.5
    ]
    spec["content_center"] = (spec["content_area"][0] +
                              spec["content_area"][2] / 2,
                              spec["content_area"][1] +
                              spec["content_area"][3] / 2)
    spec["content_top_left"] = (spec["content_area"][0],
                                spec["content_area"][1] +
                                spec["content_area"][3])


# --- Path & Edge Calculation Logic ---
def get_box_edge_intersection(center1, center2, box_coords):
    """Finds the intersection point on the edge of box_coords and nudges it outward."""
    x, y, w, h = box_coords
    cx, cy = center1
    ox, oy = center2
    dx, dy = ox - cx, oy - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9: return center1
    min_t = float('inf')
    # Check vertical edges
    if abs(dx) > 1e-9:
        t = (x - cx) / dx
        if t > 1e-9:
            iy = cy + t * dy
            if y - 1e-6 <= iy <= y + h + 1e-6: min_t = min(min_t, t)
        t = (x + w - cx) / dx
        if t > 1e-9:
            iy = cy + t * dy
            if y - 1e-6 <= iy <= y + h + 1e-6: min_t = min(min_t, t)
    # Check horizontal edges
    if abs(dy) > 1e-9:
        t = (y - cy) / dy
        if t > 1e-9:
            ix = cx + t * dx
            if x - 1e-6 <= ix <= x + w + 1e-6: min_t = min(min_t, t)
        t = (y + h - cy) / dy
        if t > 1e-9:
            ix = cx + t * dx
            if x - 1e-6 <= ix <= x + w + 1e-6: min_t = min(min_t, t)

    if min_t == float('inf') or min_t <= 1e-9: return center1
    ix = cx + min_t * dx
    iy = cy + min_t * dy
    norm = math.sqrt(dx**2 + dy**2)
    if norm < 1e-9: nudge_x, nudge_y = 0, 0
    else:
        udx, udy = dx / norm, dy / norm
        nudge_x = PACKET_SIZE * udx
        nudge_y = PACKET_SIZE * udy
    return (ix + nudge_x, iy + nudge_y)


def calculate_adjusted_path(start_comp_name, end_comp_name, waypoints):
    start_spec = COMPONENT_SPECS[start_comp_name]
    end_spec = COMPONENT_SPECS[end_comp_name]
    if len(waypoints) < 2: return waypoints
    point1 = start_spec["center"]
    point2 = waypoints[1]
    adjusted_start = get_box_edge_intersection(point1, point2,
                                               start_spec["box_coords"])
    point_end = end_spec["center"]
    point_before_end = waypoints[
        -2]  # Use the second-to-last RAW waypoint as the direction target
    adjusted_end = get_box_edge_intersection(point_end, point_before_end,
                                             end_spec["box_coords"])
    adjusted_waypoints = waypoints[:]
    adjusted_waypoints[0] = adjusted_start
    adjusted_waypoints[-1] = adjusted_end
    return adjusted_waypoints


# --- Path Definitions (Raw Waypoints) ---
RAW_PATH_DEFINITIONS = {
    ("UserInput", "MCPServer"): [
        COMPONENT_SPECS["UserInput"]["center"],
        COMPONENT_SPECS["MCPServer"]["center"]
    ],
    ("MCPServer", "MemoryStore"): [
        COMPONENT_SPECS["MCPServer"]["center"],
        (COMPONENT_SPECS["MCPServer"]["center"][0] - 0.12,
         COMPONENT_SPECS["MCPServer"]["center"][1]),  # Wider turn
        (COMPONENT_SPECS["MCPServer"]["center"][0] - 0.12,
         COMPONENT_SPECS["MemoryStore"]["center"][1]),
        COMPONENT_SPECS["MemoryStore"]["center"]
    ],
    ("MemoryStore", "MCPServer"): [
        COMPONENT_SPECS["MemoryStore"]["center"],
        (COMPONENT_SPECS["MCPServer"]["center"][0] - 0.12,
         COMPONENT_SPECS["MemoryStore"]["center"][1]),
        (COMPONENT_SPECS["MCPServer"]["center"][0] - 0.12,
         COMPONENT_SPECS["MCPServer"]["center"][1]),
        COMPONENT_SPECS["MCPServer"]["center"]
    ],
    ("MCPServer", "ToolSuite"): [
        COMPONENT_SPECS["MCPServer"]["center"],
        (COMPONENT_SPECS["MCPServer"]["center"][0] + 0.12,
         COMPONENT_SPECS["MCPServer"]["center"][1]),  # Wider turn
        (COMPONENT_SPECS["MCPServer"]["center"][0] + 0.12,
         COMPONENT_SPECS["ToolSuite"]["center"][1]),
        COMPONENT_SPECS["ToolSuite"]["center"]
    ],
    ("ToolSuite", "MCPServer"): [
        COMPONENT_SPECS["ToolSuite"]["center"],
        (COMPONENT_SPECS["MCPServer"]["center"][0] + 0.12,
         COMPONENT_SPECS["ToolSuite"]["center"][1]),
        (COMPONENT_SPECS["MCPServer"]["center"][0] + 0.12,
         COMPONENT_SPECS["MCPServer"]["center"][1]),
        COMPONENT_SPECS["MCPServer"]["center"]
    ],
    ("MCPServer", "CtxAssembler"): [
        COMPONENT_SPECS["MCPServer"]["center"],
        COMPONENT_SPECS["CtxAssembler"]["center"]
    ],
    ("CtxAssembler", "LLMEngine"): [
        COMPONENT_SPECS["CtxAssembler"]["center"],
        COMPONENT_SPECS["LLMEngine"]["center"]
    ],
    ("LLMEngine", "OutputRender"): [
        COMPONENT_SPECS["LLMEngine"]["center"],
        (COMPONENT_SPECS["LLMEngine"]["pos"][0] +
         COMPONENT_SPECS["LLMEngine"]["pos"][2] + 0.05,
         COMPONENT_SPECS["LLMEngine"]["center"][1]),
        (COMPONENT_SPECS["LLMEngine"]["pos"][0] +
         COMPONENT_SPECS["LLMEngine"]["pos"][2] + 0.05,
         COMPONENT_SPECS["OutputRender"]["center"][1]),
        (COMPONENT_SPECS["OutputRender"]["center"][0],
         COMPONENT_SPECS["OutputRender"]["center"][1]),
    ],
    ("MCPServer", "MemoryUpdater"): [
        COMPONENT_SPECS["MCPServer"]["center"],
        (COMPONENT_SPECS["MCPServer"]["center"][0] - 0.05,
         COMPONENT_SPECS["MCPServer"]["center"][1] - 0.1),  # Diagonal start
        (COMPONENT_SPECS["MemoryUpdater"]["center"][0],
         COMPONENT_SPECS["MCPServer"]["center"][1] - 0.1),
        COMPONENT_SPECS["MemoryUpdater"]["center"]
    ],
    ("MemoryUpdater", "MemoryStore"): [
        COMPONENT_SPECS["MemoryUpdater"]["center"],
        COMPONENT_SPECS["MemoryStore"]["center"]
    ],
}

# Process raw paths into adjusted paths connecting edges
ADJUSTED_PATH_DEFINITIONS = {}
for key, waypoints in RAW_PATH_DEFINITIONS.items():
    start_comp, end_comp = key
    if start_comp in COMPONENT_SPECS and end_comp in COMPONENT_SPECS:
        ADJUSTED_PATH_DEFINITIONS[key] = calculate_adjusted_path(
            start_comp, end_comp, waypoints)
    else:
        print(
            f"Warning: Component missing for path key {key}, skipping path adjustment."
        )


# --- Easing Function ---
def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2


# --- Path Point Calculation ---
def get_point_on_path(path_key, progress):
    if path_key not in ADJUSTED_PATH_DEFINITIONS: return None
    path_segments = ADJUSTED_PATH_DEFINITIONS[path_key]
    points = np.array(path_segments)
    if len(points) < 2: return tuple(points[0]) if len(points) == 1 else None
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    valid_indices = np.where(segment_lengths > 1e-9)[0]
    if len(valid_indices) == 0: return tuple(points[0])
    valid_segment_lengths = segment_lengths[valid_indices]
    total_length = np.sum(valid_segment_lengths)
    if total_length < 1e-9: return tuple(points[0])
    eased_progress = ease_in_out_sine(progress) if EASING_ENABLED else progress
    target_dist = min(1.0, max(0.0, eased_progress)) * total_length
    cumulative_length = 0
    for i, idx in enumerate(valid_indices):
        length = valid_segment_lengths[i]
        start_point_index = idx
        vector = segment_vectors[idx]
        if cumulative_length + length >= target_dist - 1e-7:
            if length < 1e-9: return tuple(points[start_point_index])
            segment_progress = min(
                1.0,
                max(0.0,
                    (target_dist - cumulative_length) / max(1e-9, length)))
            return tuple(points[start_point_index] + vector * segment_progress)
        cumulative_length += length
    final_point_index = valid_indices[-1] + 1
    if final_point_index >= len(points): final_point_index = len(points) - 1
    return tuple(points[final_point_index])


# --- Animation State Definition (Split steps 12/13 retained) ---
SIMULATION_STEPS = [
    {
        "id": 0,
        "duration": 35,
        "active": None,
        "narration": "System Idle. Waiting for user input."
    },
    {
        "id": 1,
        "duration": 55,
        "active": "MCPServer",
        "narration": "User Query Received -> Forwarding to MCP Server.",
        "actions": [("flow", "UserInput", "MCPServer", "query")]
    },
    {
        "id":
        2,
        "duration":
        35,
        "active":
        "MCPServer",
        "narration":
        "Analyzing query (needs memory context).",
        "actions": [("state", "MCPServer", "Processing Query..."),
                    ("update_text", "MCPServer",
                     "Analyzing Query...\n(Checking Memory)", 0.0)]
    },
    {
        "id":
        3,
        "duration":
        65,
        "active":
        "MemoryStore",
        "narration":
        "Querying Memory Store for relevant context.",
        "actions": [("flow", "MCPServer", "MemoryStore", "memory_query"),
                    ("state", "MemoryStore", "Searching..."),
                    ("update_text", "MemoryStore",
                     f"Searching:\n'{MEMORY_QUERY}'", 0.1, 'top', 'left'),
                    ("update_text", "MemoryStore", MEMORY_SEARCH_RESULT, 0.7,
                     'top', 'left')]
    },
    {
        "id":
        4,
        "duration":
        55,
        "active":
        "MCPServer",
        "narration":
        "Memory Store -> Responding with Context -> MCP Server.",
        "actions":
        [("flow", "MemoryStore", "MCPServer", "memory_context"),
         ("state", "MCPServer", "Receiving Data"),
         ("update_text", "MCPServer",
          f"Received Context:\n'{MEMORY_RETRIEVED}'", 0.5, 'top', 'left')]
    },
    {
        "id":
        5,
        "duration":
        35,
        "active":
        "MCPServer",
        "narration":
        "Analyzing query context (needs tools).",
        "actions": [("state", "MCPServer", "Processing Context..."),
                    ("update_text", "MCPServer",
                     "Analyzing Context...\n(Checking Tools)", 0.0)]
    },
    {
        "id":
        6,
        "duration":
        75,
        "active":
        "ToolSuite",
        "narration":
        f"Calling Tool: {TOOL_NAME}.",
        "actions":
        [("flow", "MCPServer", "ToolSuite", "tool_call"),
         ("state", "ToolSuite", "Running Tool..."),
         ("update_text", "ToolSuite", TOOL_RUNNING_TEXT, 0.1, 'top', 'left'),
         ("update_text", "ToolSuite", TOOL_RESULT, 0.7, 'top', 'left')]
    },
    {
        "id":
        7,
        "duration":
        55,
        "active":
        "MCPServer",
        "narration":
        f"{TOOL_NAME} -> Responding with Result -> MCP Server.",
        "actions": [("flow", "ToolSuite", "MCPServer", "tool_result"),
                    ("state", "MCPServer", "Receiving Data"),
                    ("update_text", "MCPServer", "Received Tool Result", 0.5)]
    },
    {
        "id":
        8,
        "duration":
        55,
        "active":
        "CtxAssembler",
        "narration":
        "MCP Server -> Sending data to Context Assembler.",
        "actions":
        [("flow", "MCPServer", "CtxAssembler", "context_data"),
         ("state", "CtxAssembler", "Assembling..."),
         ("update_text", "CtxAssembler", ASSEMBLER_INPUT, 0.1, 'top', 'left'),
         ("update_text", "CtxAssembler", ASSEMBLER_PROCESSING, 0.6)]
    },
    {
        "id":
        9,
        "duration":
        35,
        "active":
        "CtxAssembler",
        "narration":
        "Context Assembler -> Final prompt created.",
        "actions":
        [("state", "CtxAssembler", "Ready"),
         ("update_text", "CtxAssembler", ASSEMBLED_CONTEXT_LABEL, 0.0)]
    },
    {
        "id":
        10,
        "duration":
        55,
        "active":
        "LLMEngine",
        "narration":
        "Context Assembler -> Sending prompt to LLM Engine.",
        "actions": [("flow", "CtxAssembler", "LLMEngine", "final_prompt"),
                    ("state", "LLMEngine", "Receiving Prompt..."),
                    ("update_text", "LLMEngine", "Receiving Prompt...", 0.4),
                    ("update_text", "LLMEngine", LLM_PROCESSING, 0.8)]
    },
    {
        "id":
        11,
        "duration":
        90,
        "active":
        "LLMEngine",
        "narration":
        "LLM Engine -> Processing prompt...",
        "actions": [("state", "LLMEngine", "Reasoning..."),
                    ("update_text", "LLMEngine", LLM_PROCESSING, 0.0)]
    },
    # --- Split Step 12/13 ---
    {
        "id":
        12,
        "duration":
        91,
        "active":
        "LLMEngine",
        "narration":
        "LLM Engine -> Streaming response...",
        "actions": [("stream", "LLMEngine", "OutputRender", LLM_OUTPUT_STREAM,
                     "response_stream")]
    },  # Packet moves here
    {
        "id": 13,
        "duration": 39,
        "active": "OutputRender",
        "narration": "...Response Arriving -> Rendering Output.",
        "actions": [("state", "OutputRender", "Receiving Stream...")]
    },  # Target activates, text renders here
    # --- Renumbered Subsequent Steps ---
    {
        "id":
        14,
        "duration":
        35,
        "active":
        "MCPServer",
        "narration":
        "MCP Server: Evaluating need for memory update.",
        "actions": [("state", "MCPServer", "Evaluating..."),
                    ("update_text", "MCPServer",
                     "Evaluating Interaction...\n(Memory Update?)", 0.0)]
    },
    {
        "id":
        15,
        "duration":
        65,
        "active":
        "MemoryUpdater",
        "narration":
        "MCP Server -> Sending summary to Memory Updater.",
        "actions": [("flow", "MCPServer", "MemoryUpdater", "update_summary"),
                    ("state", "MemoryUpdater", "Processing Summary..."),
                    ("update_text", "MemoryUpdater",
                     f"Receiving Summary:\n'{MEMORY_UPDATE_DETAILS}'", 0.4,
                     'top', 'left')]
    },
    {
        "id":
        16,
        "duration":
        65,
        "active":
        "MemoryStore",
        "narration":
        "Memory Updater -> Storing new fact in Memory Store.",
        "actions": [("flow", "MemoryUpdater", "MemoryStore", "update_fact"),
                    ("state", "MemoryStore", "Updating..."),
                    ("update_text", "MemoryStore", "Receiving update...", 0.1,
                     'top', 'left'),
                    ("update_text", "MemoryStore",
                     MEMORY_UPDATE_CONFIRM + f"\n+ {MEMORY_UPDATE_DETAILS}",
                     0.8, 'top', 'left')]
    },
    {
        "id": 17,
        "duration": 150,
        "active": None,
        "narration": "Processing Complete. System Idle."
    },
]
TOTAL_FRAMES = sum(step["duration"] for step in SIMULATION_STEPS)

# --- Matplotlib Setup ---
plt.style.use('dark_background')
fig = plt.figure(figsize=FIG_SIZE, facecolor=FIG_BG_COLOR)
main_ax_rect = [0.02, main_ax_bottom, main_ax_width, main_ax_height]
info_ax_rect = [
    main_ax_width + 0.03, main_ax_bottom, INFO_PANEL_WIDTH, main_ax_height
]
narration_ax_rect = [
    main_ax_width + 0.03, 0.01, INFO_PANEL_WIDTH, CONTROL_PANEL_HEIGHT
]
ax_main = fig.add_axes(main_ax_rect)
ax_info = fig.add_axes(info_ax_rect)
ax_narration = fig.add_axes(narration_ax_rect)
ax_main.set_facecolor(FIG_BG_COLOR)
ax_main.set_xlim(0, 1)
ax_main.set_ylim(0, 1)
ax_main.axis('off')
ax_info.set_facecolor(INFO_BG_COLOR)
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)
ax_info.axis('off')
ax_narration.set_facecolor(FIG_BG_COLOR)
ax_narration.axis('off')
info_title = ax_info.text(0.5,
                          0.95,
                          "Component Details",
                          ha='center',
                          va='top',
                          fontsize=FONT_SIZE_LABEL + 1.5,
                          color=TEXT_LABEL_COLOR,
                          weight='bold',
                          fontfamily=FONT_FAMILY,
                          wrap=True)
info_text = ax_info.text(0.05,
                         0.88,
                         "Click on a component\nin the simulation.",
                         ha='left',
                         va='top',
                         fontsize=FONT_SIZE_INFO,
                         color=INFO_TEXT_COLOR,
                         wrap=True,
                         fontfamily=FONT_FAMILY,
                         linespacing=1.4,
                         clip_on=True)
narration_bar = ax_narration.text(0.5,
                                  0.5,
                                  "",
                                  ha='center',
                                  va='center',
                                  fontsize=FONT_SIZE_NARRATION,
                                  color=TEXT_COLOR,
                                  wrap=True,
                                  fontfamily=FONT_FAMILY,
                                  clip_on=True)
narration_bar.set_bbox(
    dict(facecolor=NARRATION_BG,
         alpha=0.85,
         boxstyle=f'round,pad=0.5,rounding_size={CORNER_RADIUS*2}'))

# --- Global State Variables ---
component_patches = {}
component_texts = {}
component_labels = {}
component_glows = {}
path_lines = {}
component_artists = {}
active_packets = []
active_path_key = None
streaming_text_info = None
component_states = {}
current_frame_index = 0
current_step_index = -1
frame_in_current_step = 0
last_active_component = None
ani = None
is_paused = False
animation_speed_factor = 1.0
effective_frame_interval = DEFAULT_FRAME_INTERVAL_MS

# --- Control Buttons ---
control_area_y_bottom = 0.01
control_area_height = CONTROL_PANEL_HEIGHT
btn_h = control_area_height * 0.7
btn_y = control_area_y_bottom + (control_area_height - btn_h) / 2
btn_w_large = 0.08
btn_w_small = 0.04
spacing = 0.01
margin_right = 0.01
controls_right_edge = main_ax_rect[0] + main_ax_rect[2]
reset_rect = [
    controls_right_edge - margin_right - btn_w_large, btn_y, btn_w_large, btn_h
]
pause_rect = [reset_rect[0] - spacing - btn_w_large, btn_y, btn_w_large, btn_h]
speed_1x_rect = [
    pause_rect[0] - spacing - btn_w_small, btn_y, btn_w_small, btn_h
]
speed_2x_rect = [
    speed_1x_rect[0] - spacing - btn_w_small, btn_y, btn_w_small, btn_h
]
speed_05x_rect = [
    speed_2x_rect[0] - spacing - btn_w_small, btn_y, btn_w_small, btn_h
]
reset_ax = fig.add_axes(reset_rect)  # type: ignore
pause_ax = fig.add_axes(pause_rect)  # type: ignore
speed_1x_ax = fig.add_axes(speed_1x_rect)  # type: ignore
speed_2x_ax = fig.add_axes(speed_2x_rect)  # type: ignore
speed_05x_ax = fig.add_axes(speed_05x_rect)  # type: ignore
reset_button = Button(reset_ax,
                      'Reset',
                      color=BUTTON_BG,
                      hovercolor=BUTTON_HOVER_BG)
pause_button = Button(pause_ax,
                      '❚❚ Pause',
                      color=BUTTON_BG,
                      hovercolor=BUTTON_HOVER_BG)
speed_1x_button = Button(speed_1x_ax,
                         '1x',
                         color=BUTTON_BG,
                         hovercolor=BUTTON_HOVER_BG)
speed_2x_button = Button(speed_2x_ax,
                         '2x',
                         color=BUTTON_BG,
                         hovercolor=BUTTON_HOVER_BG)
speed_05x_button = Button(speed_05x_ax,
                          '0.5x',
                          color=BUTTON_BG,
                          hovercolor=BUTTON_HOVER_BG)
all_buttons = [
    reset_button, pause_button, speed_1x_button, speed_2x_button,
    speed_05x_button
]
for btn in all_buttons:
    btn.label.set_color(BUTTON_TEXT_COLOR)
    btn.label.set_fontfamily(FONT_FAMILY)
    btn.label.set_fontsize(FONT_SIZE_LABEL - 1)
speed_1x_button.ax.patch.set_edgecolor(ACTIVE_BORDER_COLOR)
speed_1x_button.ax.patch.set_linewidth(1.5)  # Initial highlight


# --- Drawing Functions & State ---
def draw_components():
    """Clears and redraws all static simulation components and paths."""
    global component_artists
    # Clear existing artists safely
    for name, artist_list in list(component_artists.items()):
        for artist in artist_list:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass  # Ignore if already removed or invalid
    for key, line in list(path_lines.items()):
        try:
            line.remove()
        except (ValueError, AttributeError):
            pass
    for packet_info in list(active_packets):
        try:
            if packet_info.get('artist'): packet_info['artist'].remove()
        except (ValueError, AttributeError):
            pass

    # Clear dictionaries
    component_artists.clear()
    path_lines.clear()
    active_packets.clear()
    component_patches.clear()
    component_texts.clear()
    component_labels.clear()
    component_glows.clear()
    component_states.clear()

    # Draw paths
    for path_key, adjusted_waypoints in ADJUSTED_PATH_DEFINITIONS.items():
        path_points = np.array(adjusted_waypoints)
        if len(path_points) >= 2:
            line = lines.Line2D(path_points[:, 0],
                                path_points[:, 1],
                                color=PATH_COLOR,
                                linewidth=PATH_LINEWIDTH,
                                alpha=INACTIVE_PATH_ALPHA,
                                linestyle='-',
                                zorder=1,
                                transform=ax_main.transAxes)
            ax_main.add_line(line)
            path_lines[path_key] = line

    # Draw components
    for name, spec in COMPONENT_SPECS.items():
        component_artists[name] = []
        x, y, w, h = spec["pos"]
        boxstyle = f"round,pad={COMPONENT_PADDING*0.5},rounding_size={CORNER_RADIUS}"

        # Glow patch (behind main patch)
        glow_patch = patches.FancyBboxPatch((x - 0.006, y - 0.006),
                                            w + 0.012,
                                            h + 0.012,
                                            boxstyle=boxstyle,
                                            edgecolor=GLOW_COLOR,
                                            facecolor='none',
                                            linewidth=4,
                                            alpha=0.0,
                                            zorder=2,
                                            transform=ax_main.transAxes)
        ax_main.add_patch(glow_patch)
        component_glows[name] = glow_patch
        component_artists[name].append(glow_patch)

        # Main component patch (clickable)
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=boxstyle,
            edgecolor=BORDER_COLOR,
            facecolor=COMPONENT_BG_COLOR,
            linewidth=BORDER_WIDTH,
            clip_on=False,
            zorder=3,
            transform=ax_main.transAxes,
            picker=True,  # Enable picking
            alpha=INACTIVE_COMPONENT_ALPHA)
        component_patches[name] = rect  # Map component name to its patch
        ax_main.add_patch(rect)
        component_patches[name] = rect
        component_artists[name].append(rect)

        # Label
        label_y = y + h - LABEL_HEIGHT / 2
        label = ax_main.text(x + w / 2,
                             label_y,
                             spec["label"],
                             ha='center',
                             va='center',
                             fontsize=FONT_SIZE_LABEL,
                             color=TEXT_LABEL_COLOR,
                             weight=FONT_WEIGHT_LABEL,
                             zorder=4,
                             fontfamily=FONT_FAMILY,
                             transform=ax_main.transAxes)
        component_labels[name] = label
        component_artists[name].append(label)

        # Content Text Area
        content_x, content_y = spec["content_center"]
        content_bbox = Bbox.from_bounds(
            *spec["content_area"])  # Bounding box for text clipping
        content_text = ax_main.text(
            content_x,
            content_y,
            "",
            ha='center',
            va='center',
            fontsize=FONT_SIZE_IO,
            color=TEXT_COLOR,
            wrap=True,
            zorder=4,
            fontfamily=FONT_FAMILY,
            transform=ax_main.transAxes,
            linespacing=1.3,
            clip_box=content_bbox)  # <<< Use clip_box to contain text visually
        component_texts[name] = content_text
        component_artists[name].append(content_text)
        component_states[name] = "Idle"  # Initial state

    reset_component_texts()  # Set initial text values


def reset_component_texts():
    """Sets initial text content and alignment for specific components."""
    init_va, init_ha = 'top', 'left'
    center_va, center_ha = 'center', 'center'
    if "UserInput" in component_texts:
        component_texts["UserInput"].set_text(f'"{USER_QUERY}"')
        set_text_alignment(component_texts["UserInput"], "UserInput",
                           center_va, center_ha)
    if "MemoryStore" in component_texts:
        component_texts["MemoryStore"].set_text(MEMORY_CONTENT_INITIAL)
        set_text_alignment(component_texts["MemoryStore"], "MemoryStore",
                           init_va, init_ha)
    if "ToolSuite" in component_texts:
        component_texts["ToolSuite"].set_text(
            f"Available:\n- {TOOL_NAME}\n- WebSearch...")
        set_text_alignment(component_texts["ToolSuite"], "ToolSuite", init_va,
                           init_ha)
    if "OutputRender" in component_texts:
        component_texts["OutputRender"].set_text("Awaiting response...")
        set_text_alignment(component_texts["OutputRender"], "OutputRender",
                           center_va, center_ha)
    for name in ["MCPServer", "CtxAssembler", "LLMEngine", "MemoryUpdater"]:
        if name in component_texts:
            component_texts[name].set_text("Idle")
            set_text_alignment(component_texts[name], name, center_va,
                               center_ha)
            component_states[name] = "Idle"


def set_text_alignment(text_obj, comp_name, va, ha):
    """Helper to set text alignment and position based on component spec."""
    if comp_name not in COMPONENT_SPECS: return
    spec = COMPONENT_SPECS[comp_name]
    text_obj.set_va(va)
    text_obj.set_ha(ha)
    # Calculate position based on alignment
    x_pos, y_pos = spec["content_center"]
    if va == 'top':
        y_pos = spec["content_top_left"][1] - COMPONENT_PADDING * 0.5
    elif va == 'bottom':
        y_pos = spec["content_area"][1] + COMPONENT_PADDING * 0.5
    if ha == 'left': x_pos = spec["content_area"][0] + COMPONENT_PADDING * 0.1
    elif ha == 'right':
        x_pos = spec["content_area"][0] + spec["content_area"][
            2] - COMPONENT_PADDING * 0.1
    # Apply position safely
    if not (np.isnan(x_pos) or np.isnan(y_pos)):
        try:
            text_obj.set_position((x_pos, y_pos))
        except Exception as e:
            print(
                f"Warning: Could not set position ({x_pos}, {y_pos}) for {comp_name}: {e}"
            )


def format_info_text(raw_text):
    """Rudimentary formatter for info panel text."""
    lines = raw_text.split('\n')
    formatted_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('**') and '**:' in stripped_line:
            parts = stripped_line.split('**:', 1)
            if len(parts) == 2:
                bold_part = parts[0].replace('**', '').strip() + ':'
                rest_part = parts[1]
                indent = line[:len(line) - len(line.lstrip(' '))]
                formatted_lines.append(f"{indent}{bold_part}{rest_part}")
            else:
                formatted_lines.append(line.replace('**', ''))  # Fallback
        elif stripped_line.startswith('- '):
            indent = line[:len(line) - len(line.lstrip(' '))]
            formatted_lines.append(f"{indent}  • {stripped_line[2:]}")
        else:
            formatted_lines.append(line)
    return '\n'.join(formatted_lines)


# --- Animation Logic ---
def update(frame_num):
    """Main animation update function, called for each frame."""
    global current_step_index, frame_in_current_step, last_active_component, \
           active_packets, active_path_key, streaming_text_info, current_frame_index, \
           is_paused

    # Handle Pausing
    if is_paused:
        # Return all potentially visible artists to keep the display static
        artists_to_draw = [narration_bar, info_title, info_text]
        for artist_list in component_artists.values():
            artists_to_draw.extend(a for a in artist_list if a is not None)
        artists_to_draw.extend(l for l in path_lines.values() if l is not None)
        artists_to_draw.extend(p['artist'] for p in active_packets
                               if p.get('artist') is not None)
        for btn in all_buttons:
            artists_to_draw.extend(a for a in btn.ax.get_children()
                                   if a is not None)
        return [a for a in artists_to_draw if a is not None]

    # Update global frame index
    current_frame_index = frame_num

    # Determine current step
    cumulative_frames = 0
    target_step_index = -1
    target_frame_in_step = 0
    for idx, step in enumerate(SIMULATION_STEPS):
        step_duration = step["duration"]
        if step_duration <= 0: continue  # Skip steps with no duration
        if current_frame_index < cumulative_frames + step_duration:
            target_step_index = idx
            target_frame_in_step = current_frame_index - cumulative_frames
            break
        cumulative_frames += step_duration
    else:  # If loop finishes, we are past the end
        target_step_index = len(SIMULATION_STEPS) - 1
        target_frame_in_step = SIMULATION_STEPS[target_step_index][
            "duration"] - 1

    # Handle end of animation
    if current_frame_index >= TOTAL_FRAMES:
        if current_frame_index == TOTAL_FRAMES:  # Perform cleanup only once
            # Deactivate last component
            if last_active_component and last_active_component in component_patches:
                component_patches[last_active_component].set_edgecolor(
                    BORDER_COLOR)
                component_patches[last_active_component].set_linewidth(
                    BORDER_WIDTH)
                component_patches[last_active_component].set_alpha(
                    INACTIVE_COMPONENT_ALPHA)
                if last_active_component in component_glows:
                    component_glows[last_active_component].set_alpha(0.0)
            # Deactivate last path
            if active_path_key and active_path_key in path_lines:
                try:
                    path_lines[active_path_key].set_color(PATH_COLOR)
                    path_lines[active_path_key].set_linewidth(PATH_LINEWIDTH)
                    path_lines[active_path_key].set_alpha(INACTIVE_PATH_ALPHA)
                    path_lines[active_path_key].set_zorder(1)
                except KeyError:
                    pass
            # Remove any stray packets
            for p_info in list(active_packets):
                try:
                    artist = p_info.get("artist")
                    if artist: artist.remove()
                except (ValueError, AttributeError):
                    pass
            active_packets.clear()
            narration_bar.set_text("Simulation Complete.")

        # Keep displaying final state
        artists_to_draw = [narration_bar, info_title, info_text]
        for artist_list in component_artists.values():
            artists_to_draw.extend(a for a in artist_list if a is not None)
        artists_to_draw.extend(l for l in path_lines.values() if l is not None)
        artists_to_draw.extend(
            p['artist'] for p in active_packets
            if p.get('artist') is not None)  # Should be empty
        for btn in all_buttons:
            artists_to_draw.extend(a for a in btn.ax.get_children()
                                   if a is not None)
        return [a for a in artists_to_draw if a is not None]

    # Detect new step
    is_new_step = (target_step_index != current_step_index)
    current_step_index = target_step_index
    frame_in_current_step = target_frame_in_step

    if current_step_index < 0: return []  # Should not happen

    step_info = SIMULATION_STEPS[current_step_index]
    step_duration = step_info["duration"]
    safe_step_duration = max(1, step_duration)

    # --- Step Initialization Logic ---
    if is_new_step:
        new_active_comp = step_info.get("active")
        path_reused_in_current_step = False
        current_step_path_key = None
        for action in step_info.get("actions", []):  # Check path reuse
            if action[0] in ["flow", "stream"]:
                current_step_path_key = (action[1], action[2])
                if current_step_path_key == active_path_key:
                    path_reused_in_current_step = True
                break

        # Reset previous path style only if not reused
        if active_path_key and not path_reused_in_current_step and active_path_key in path_lines:
            try:
                path_lines[active_path_key].set_color(PATH_COLOR)
                path_lines[active_path_key].set_linewidth(PATH_LINEWIDTH)
                path_lines[active_path_key].set_alpha(INACTIVE_PATH_ALPHA)
                path_lines[active_path_key].set_zorder(1)
            except KeyError:
                pass
            active_path_key = None  # Clear tracker if path changed

        # Reset previous component style if different
        if last_active_component and last_active_component != new_active_comp:
            if last_active_component in component_patches:
                component_patches[last_active_component].set_edgecolor(
                    BORDER_COLOR)
                component_patches[last_active_component].set_linewidth(
                    BORDER_WIDTH)
                component_patches[last_active_component].set_alpha(
                    INACTIVE_COMPONENT_ALPHA)
                if last_active_component in component_glows:
                    component_glows[last_active_component].set_alpha(0.0)

        # Set new active component style
        if new_active_comp and new_active_comp in component_patches:
            component_patches[new_active_comp].set_edgecolor(
                ACTIVE_BORDER_COLOR)
            component_patches[new_active_comp].set_linewidth(
                ACTIVE_BORDER_WIDTH)
            component_patches[new_active_comp].set_alpha(1.0)
            last_active_component = new_active_comp
        elif not new_active_comp and last_active_component:  # Deactivate if step has no active comp
            if last_active_component in component_patches:
                component_patches[last_active_component].set_edgecolor(
                    BORDER_COLOR)
                component_patches[last_active_component].set_linewidth(
                    BORDER_WIDTH)
                component_patches[last_active_component].set_alpha(
                    INACTIVE_COMPONENT_ALPHA)
                if last_active_component in component_glows:
                    component_glows[last_active_component].set_alpha(0.0)
            last_active_component = None

        # Update Narration
        narration_bar.set_text(
            f"Step {step_info['id']}: {step_info['narration']}")

        # Process actions for the new step
        actions = step_info.get("actions", [])
        for action in actions:
            action_type = action[0]
            if action_type == "flow" or action_type == "stream":
                _, start_comp, end_comp, data_type_or_parts, *rest = action
                data_type = data_type_or_parts if action_type == "flow" else rest[
                    0] if rest else "default"
                text_parts = data_type_or_parts if action_type == "stream" else None
                path_key = (start_comp, end_comp)

                if path_key in path_lines and path_key in ADJUSTED_PATH_DEFINITIONS:
                    active_path_key = path_key  # Set current active path
                    # Style path
                    path_lines[path_key].set_color(ACTIVE_PATH_COLOR)
                    path_lines[path_key].set_linewidth(ACTIVE_PATH_LINEWIDTH)
                    path_lines[path_key].set_alpha(1.0)
                    path_lines[path_key].set_zorder(9)
                    # Create packet
                    start_pos = get_point_on_path(path_key, 0.0)
                    if start_pos:
                        packet_color = PACKET_COLORS.get(
                            data_type, PACKET_COLORS["default"])
                        packet = patches.Circle(start_pos,
                                                radius=PACKET_SIZE,
                                                color=packet_color,
                                                zorder=10,
                                                transform=ax_main.transAxes,
                                                alpha=0.95)
                        ax_main.add_patch(packet)
                        active_packets.append({
                            "artist":
                            packet,
                            "path_key":
                            path_key,
                            "progress":
                            0.0,
                            "data_type":
                            data_type,
                            "step_start_frame":
                            current_frame_index,
                            "step_duration":
                            safe_step_duration
                        })
                    else:
                        print(
                            f"Warning: Could not get start pos for packet on {path_key}."
                        )
                else:
                    print(
                        f"Warning: Path key {path_key} missing for '{action_type}'."
                    )

                # Setup streaming state if this is a stream action with text parts
                if action_type == "stream" and text_parts:
                    num_parts = len(text_parts)
                    # Determine which step handles the actual processing/rendering
                    processing_step_index = 13 if current_step_index == 12 else current_step_index
                    processing_step_duration = max(
                        1, SIMULATION_STEPS[processing_step_index]["duration"])
                    frames_per_part = max(
                        1, processing_step_duration // max(1, num_parts))

                    streaming_text_info = {
                        "target": end_comp,
                        "parts": text_parts,
                        "current_part_index": 0,
                        "accumulated_text": "",
                        "frames_per_part": frames_per_part,
                        "data_type": data_type,
                        "processing_step_index": processing_step_index
                    }
                    # Prepare target text area
                    if end_comp in component_texts:
                        component_texts[end_comp].set_text("")  # Clear
                        set_text_alignment(component_texts[end_comp], end_comp,
                                           'top', 'left')  # Align

            elif action_type == "state":
                _, comp_name, state_text = action
                if comp_name in component_states:
                    component_states[comp_name] = state_text
                if comp_name in component_texts:
                    # Set text immediately, clip_box should handle overflow prevention
                    set_text_alignment(component_texts[comp_name], comp_name,
                                       'center', 'center')
                    component_texts[comp_name].set_text(state_text)

    # --- Animate within the current step ---
    step_progress = frame_in_current_step / max(
        1, safe_step_duration - 1) if safe_step_duration > 1 else 1.0
    step_progress = min(1.0, max(0.0, step_progress))

    # Animate Packets
    packets_to_remove_indices = []
    for i, packet_info in enumerate(active_packets):
        # Calculate progress based on packet's own step timing
        pkt_step_duration = packet_info.get("step_duration",
                                            safe_step_duration)
        pkt_start_frame = packet_info.get(
            "step_start_frame", current_frame_index - frame_in_current_step)
        frames_elapsed_for_packet = current_frame_index - pkt_start_frame
        packet_absolute_progress = frames_elapsed_for_packet / max(
            1, pkt_step_duration - 1) if pkt_step_duration > 1 else 1.0
        packet_absolute_progress = min(1.0, max(0.0, packet_absolute_progress))

        # Visual progress scaled by the factor (determines *when* it arrives visually)
        packet_visual_progress = min(
            1.0, packet_absolute_progress / max(0.01, PACKET_DURATION_FACTOR))
        eased_packet_progress = ease_in_out_sine(
            packet_visual_progress
        ) if EASING_ENABLED else packet_visual_progress

        path_key = packet_info["path_key"]
        artist = packet_info.get('artist')

        if artist is None or path_key not in ADJUSTED_PATH_DEFINITIONS:
            if i not in packets_to_remove_indices:
                packets_to_remove_indices.append(i)
            continue

        current_pos = get_point_on_path(path_key, eased_packet_progress)
        if current_pos:
            try:
                artist.center = tuple(current_pos)
            except (ValueError, AttributeError):
                if i not in packets_to_remove_indices:
                    packets_to_remove_indices.append(i)
                    continue
        else:  # Failed to get position
            if i not in packets_to_remove_indices:
                packets_to_remove_indices.append(i)
                continue

        packet_info[
            "progress"] = eased_packet_progress  # Store current visual progress

        # Fade and remove based on visual progress completion
        fade_start_progress = 0.95  # Start fading near the visual end
        if packet_visual_progress >= fade_start_progress:
            fade = max(
                0, 1.0 - (packet_visual_progress - fade_start_progress) /
                max(1e-6, (1.0 - fade_start_progress)))
            try:
                artist.set_alpha(fade * 0.95)
            except (ValueError, AttributeError):
                pass
            # Mark for removal *when visual travel is complete*
            if packet_visual_progress >= 1.0:
                if i not in packets_to_remove_indices:
                    packets_to_remove_indices.append(i)
        else:  # Ensure full alpha before fade starts
            try:
                artist.set_alpha(0.95)
            except (ValueError, AttributeError):
                pass

    # Remove completed packets
    if packets_to_remove_indices:
        for index in sorted(list(set(packets_to_remove_indices)),
                            reverse=True):
            if 0 <= index < len(active_packets):
                try:
                    removed_artist = active_packets[index].get("artist")
                    if removed_artist: removed_artist.remove()
                except (ValueError, AttributeError, IndexError):
                    pass
                try:
                    del active_packets[index]
                except IndexError:
                    pass

    # Animate Active Component Glow
    active_comp_for_glow = step_info.get("active")
    if active_comp_for_glow and active_comp_for_glow in component_glows:
        pulse_phase = math.sin(current_frame_index * GLOW_PULSE_SPEED +
                               current_step_index)
        glow_alpha = max(0, 0.5 + 0.4 * pulse_phase)
        component_glows[active_comp_for_glow].set_alpha(glow_alpha)

    # Handle Scheduled Text Updates (within the current step)
    current_step_actions = step_info.get("actions", [])
    for action in current_step_actions:
        action_type = action[0]
        if action_type == "update_text":
            _, comp_name, text, delay_ratio, *align = action
            if step_progress >= delay_ratio and comp_name in component_texts:
                text_obj = component_texts[comp_name]
                if text_obj.get_text() != text:  # Avoid redundant updates
                    va = align[0] if len(align) > 0 else 'center'
                    ha = align[1] if len(align) > 1 else 'center'
                    text_obj.set_text(text)  # clip_box handles overflow
                    set_text_alignment(text_obj, comp_name, va,
                                       ha)  # Set alignment/position

    # Handle Active Text Streaming (if conditions are met)
    if streaming_text_info and current_step_index == streaming_text_info.get(
            "processing_step_index"):
        target_comp = streaming_text_info["target"]
        if target_comp in component_texts:
            target_text_obj = component_texts[target_comp]
            frames_per = streaming_text_info["frames_per_part"]
            current_idx = streaming_text_info["current_part_index"]
            parts = streaming_text_info["parts"]
            frame_in_processing_step = frame_in_current_step

            safe_frames_per = max(1, frames_per)
            expected_part_index = frame_in_processing_step // safe_frames_per

            # Add parts incrementally
            while current_idx <= expected_part_index and current_idx < len(
                    parts):
                streaming_text_info["accumulated_text"] += parts[current_idx]
                streaming_text_info["current_part_index"] += 1
                current_idx += 1

            # Update display text (with potential cursor)
            current_display_text = streaming_text_info["accumulated_text"]
            cursor_visible = (current_idx < len(parts)
                              )  # Cursor needed if not all parts added

            # Determine if cursor should blink on or off
            blink_speed = 12
            show_cursor_now = cursor_visible and (
                frame_in_processing_step % blink_speed >= blink_speed // 2)

            final_text_to_display = current_display_text + (
                "▌" if show_cursor_now else "")

            # Update text object only if the visual text needs to change
            if target_text_obj.get_text() != final_text_to_display:
                target_text_obj.set_text(final_text_to_display)

            # Set final text without cursor when streaming is fully complete
            if not cursor_visible and target_text_obj.get_text(
            ) != FINAL_RESPONSE:
                target_text_obj.set_text(FINAL_RESPONSE)
                set_text_alignment(target_text_obj, target_comp, 'top',
                                   'left')  # Re-align final response

    # --- Collect Artists for Drawing ---
    artists_to_draw = [narration_bar, info_title, info_text]
    for artist_list in component_artists.values():
        artists_to_draw.extend(a for a in artist_list if a is not None)
    artists_to_draw.extend(l for l in path_lines.values() if l is not None)
    artists_to_draw.extend(p['artist'] for p in active_packets
                           if p.get('artist') is not None)
    for btn in all_buttons:
        artists_to_draw.extend(a for a in btn.ax.get_children()
                               if a is not None)
    return [a for a in artists_to_draw if a is not None]


# --- Event Handling ---
def on_click(event):
    """Handles clicks on simulation components (Info Panel Update)."""
    if event.inaxes != ax_main or event.button != 1: return

    clicked_comp = None
    artist = getattr(event, 'artist', None)
    if artist and isinstance(artist, patches.FancyBboxPatch) and hasattr(
            artist, 'component_name'):
        if artist in component_patches.values(
        ):  # Check it's a main component patch
            contains, _ = artist.contains(event)
            if contains:
                clicked_comp = next(
                    (name for name, patch in component_patches.items()
                     if patch is artist), None)

    if clicked_comp and clicked_comp in COMPONENT_DETAILS:
        detail_raw = COMPONENT_DETAILS[clicked_comp]
        detail_formatted = format_info_text(detail_raw)
        current_state = component_states.get(clicked_comp, "Idle")
        current_content = component_texts[clicked_comp].get_text().rstrip(
            "▌") if clicked_comp in component_texts else ""

        state_info = f"\n\n**Current Status:** {current_state}"
        clean_content = current_content.strip()
        if clean_content and current_state != "Idle" and not any(
                clean_content.startswith(s)
                for s in ["Idle", "Awaiting response", "Available:"]):
            max_len = 150
            display_content = clean_content[:max_len] + (
                '...' if len(clean_content) > max_len else '')
            state_info += f"\n**Content Snippet:**\n  {display_content}"

        final_info = f"--- {clicked_comp} ---\n{detail_formatted}\n{format_info_text(state_info)}"
        info_text.set_text(final_info)
        fig.canvas.draw_idle()


def stop_animation():
    global ani
    if ani and getattr(ani, 'event_source', None) and hasattr(
            ani.event_source, 'stop'):
        try:
            ani.event_source.stop()
        except Exception:
            pass  # Ignore errors on stop


def start_animation(start_frame=0):
    global ani, effective_frame_interval, is_paused
    stop_animation()
    effective_frame_interval = max(
        1, int(DEFAULT_FRAME_INTERVAL_MS / animation_speed_factor))
    end_frame = TOTAL_FRAMES + 10
    frame_iterable = range(start_frame, end_frame)
    ani = animation.FuncAnimation(fig,
                                  update,
                                  frames=frame_iterable,
                                  interval=effective_frame_interval,
                                  blit=False,
                                  repeat=False,
                                  cache_frame_data=False)
    global animation_instance
    animation_instance = ani
    is_paused = False
    pause_button.label.set_text('❚❚ Pause')
    fig.canvas.draw_idle()


def reset_simulation(event=None):
    global current_frame_index, current_step_index, frame_in_current_step, last_active_component, \
           active_packets, active_path_key, streaming_text_info, is_paused
    print("Resetting simulation...")
    stop_animation()
    # Reset state variables
    current_frame_index = 0
    current_step_index = -1
    frame_in_current_step = 0
    last_active_component = None
    active_path_key = None
    streaming_text_info = None
    is_paused = False
    # Reset UI
    info_text.set_text("Click on a component\nin the simulation.")
    narration_bar.set_text("")  # update(0) will set it
    pause_button.label.set_text('❚❚ Pause')
    # Redraw and restart
    draw_components()
    update_speed_buttons()
    start_animation(start_frame=0)
    print("Simulation Reset Complete.")


def toggle_pause(event=None):
    global is_paused, ani, current_frame_index
    is_paused = not is_paused
    if is_paused:
        pause_button.label.set_text('▶ Play')
        stop_animation()
        print(f"Animation Paused at frame {current_frame_index}")
    else:
        pause_button.label.set_text('❚❚ Pause')
        resume_frame = current_frame_index  # Resume from the exact frame
        print(f"Animation Resuming from frame {resume_frame}")
        start_animation(start_frame=resume_frame)
    fig.canvas.draw_idle()


def update_speed_buttons():
    buttons = {
        0.5: speed_05x_button,
        1.0: speed_1x_button,
        2.0: speed_2x_button
    }
    for speed, btn in buttons.items():
        try:
            patch = btn.ax.patch
            is_active = math.isclose(speed, animation_speed_factor)
            patch.set_edgecolor(
                ACTIVE_BORDER_COLOR if is_active else BUTTON_BG)
            patch.set_linewidth(1.5 if is_active else 1.0)
        except AttributeError:
            pass


def change_speed(event=None, factor=1.0):
    global animation_speed_factor, is_paused, current_frame_index, effective_frame_interval
    if math.isclose(factor, animation_speed_factor): return
    print(f"Changing speed to {factor}x")
    animation_speed_factor = factor
    effective_frame_interval = max(
        1, int(DEFAULT_FRAME_INTERVAL_MS / animation_speed_factor))
    update_speed_buttons()
    if not is_paused:
        resume_frame = current_frame_index  # Restart from current frame with new speed
        print(f"Restarting animation at frame {resume_frame} with new speed.")
        start_animation(start_frame=resume_frame)
    else:
        print("Speed changed while paused. Will apply on resume.")
    fig.canvas.draw_idle()


# --- Connect Events ---
reset_button.on_clicked(reset_simulation)
pause_button.on_clicked(toggle_pause)
speed_1x_button.on_clicked(lambda e: change_speed(e, factor=1.0))
speed_2x_button.on_clicked(lambda e: change_speed(e, factor=2.0))
speed_05x_button.on_clicked(lambda e: change_speed(e, factor=0.5))
fig.canvas.mpl_connect('pick_event', on_click)  # Connect the click handler

# --- Initial Setup & Start ---
print("Initial draw and animation start...")
reset_simulation()
plt.show()
print("Simulation window closed.")
