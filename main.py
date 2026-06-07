import asyncio
import threading
import urllib.request
import numpy as np
from PIL import Image
import streamlit as st
import spade
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import time

if "room_data" not in st.session_state:
    st.session_state.room_data = {
        "room_type": "Living Room",  
        "base_budget": 5000.0,
        "budget": 5000,
        "floor_cost": 0,
        "furniture_cost": 0,
        "style_score": 0,
        "flooring": "None",
        "wall_color": "#FFFFFF",
        "width": 5.0,
        "height": 5.0,
        "openings": [], 
        "furniture": [], 
        "user_style_preference": "Minimalist",
        "alerts": [],
        "history": ["System initialized."]
    }

PRICING = {
    "None": 0, "Hardwood": 800, "Marble": 3000, "Carpet": 500
}

def log_agent_action(agent_name, message):
    time_stamp = time.strftime("%H:%M:%S")
    st.session_state.room_data["history"].append(f"[{time_stamp}]  {agent_name}: {message}")

def check_overlap(wall, pos, size, exclude_idx=None):
    new_min = pos - (size / 2)
    new_max = pos + (size / 2)
    for idx, op in enumerate(st.session_state.room_data["openings"]):
        if exclude_idx is not None and idx == exclude_idx: continue
        if op["wall"] == wall:
            exist_min = op["pos"] - (op["size"] / 2)
            exist_max = op["pos"] + (op["size"] / 2)
            if new_min < exist_max and new_max > exist_min: return True
    return False

def get_rotated_corners(x, y, w, h, angle_deg):
    rad = np.radians(angle_deg)
    local_corners = np.array([
        [-w/2, -h/2], [w/2, -h/2], 
        [w/2, h/2],  [-w/2, h/2]
    ])
    rot_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ])
    return (local_corners @ rot_matrix.T) + np.array([x, y])

def check_wall_collision(item, new_x, new_y, new_angle, room_w, room_h):
    global_corners = get_rotated_corners(new_x, new_y, item["w_m"], item["h_m"], new_angle)
    min_x, min_y = np.min(global_corners, axis=0)
    max_x, max_y = np.max(global_corners, axis=0)
    return min_x <= 0.0 or max_x >= room_w or min_y <= 0.0 or max_y >= room_h

def check_item_to_item_collision(moving_idx, target_x, target_y, target_angle):
    furniture_list = st.session_state.room_data["furniture"]
    moving_item = furniture_list[moving_idx]
    
    box1 = get_rotated_corners(target_x, target_y, moving_item["w_m"], moving_item["h_m"], target_angle)
    
    for idx, other in enumerate(furniture_list):
        if idx == moving_idx:
            continue
            
        box2 = get_rotated_corners(other["x"], other["y"], other["w_m"], other["h_m"], other.get("angle", 0))
        
        if polygons_overlap(box1, box2):
            return other["name"]
    return None

def polygons_overlap(poly1, poly2):
    for poly in [poly1, poly2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            
            proj1 = [np.dot(v, normal) for v in poly1]
            proj2 = [np.dot(v, normal) for v in poly2]
            
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False
    return True

class ArchitectAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internal_room_model = {
            "width": 0.0,
            "height": 0.0,
            "openings_count": 0
        }

    class TrackStructuralModels(CyclicBehaviour):
        async def run(self):
            state = st.session_state.room_data
            
            if (self.agent.internal_room_model["width"] != state["width"] or 
                self.agent.internal_room_model["height"] != state["height"] or 
                self.agent.internal_room_model["openings_count"] != len(state["openings"])):
                
                self.agent.internal_room_model["width"] = state["width"]
                self.agent.internal_room_model["height"] = state["height"]
                self.agent.internal_room_model["openings_count"] = len(state["openings"])
                log_agent_action("Architect Agent", f"Internal tracking model synchronized to dimensions: {state['width']}m x {state['height']}m.")

            for op in state["openings"]:
                limit = state["width"] if op["wall"] in [0, 2] else state["height"]
                if op["pos"] > limit:
                    alert = f" **Architect Agent**: Opening layout coordinates drift outside room bounds ({limit}m)!"
                    if alert not in st.session_state.room_data["alerts"]:
                        st.session_state.room_data["alerts"].append(alert)
            await asyncio.sleep(2.0)

    async def setup(self):
        self.add_behaviour(self.TrackStructuralModels())

class StructuralAgent(Agent):
    class EvaluateMaterials(CyclicBehaviour):
        async def run(self):
            state = st.session_state.room_data
            floor = state["flooring"]
            room = state["room_type"]
            wall_color = state["wall_color"]
            
            if floor == "Marble":
                alert = " **Structural Agent**: Marble is expensive and high maintenance!"
                if alert not in st.session_state.room_data["alerts"]:
                    st.session_state.room_data["alerts"].append(alert)
            if room == "Bedroom" and floor == "Marble":
                alert = " **Structural Agent**: Marble feels cold for bedrooms. Consider Hardwood."
                if alert not in st.session_state.room_data["alerts"]:
                    st.session_state.room_data["alerts"].append(alert)
            if wall_color.upper() in ["#000000", "#1E1E1E", "#333333"]:
                alert = " **Structural Agent**: Dark walls absorb light! Ensure ample window apertures."
                if alert not in st.session_state.room_data["alerts"]:
                    st.session_state.room_data["alerts"].append(alert)
            
            log_agent_action("Structural Agent", f"Reflex check completed for floor '{floor}'.")
            await asyncio.sleep(3.0)

    async def setup(self):
        self.add_behaviour(self.EvaluateMaterials())

class DesignerAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.catalog = {
            "Living Room": [
                {"name": "Contemporary Dining Set", "cost": 1400, "style": 45, "w_m": 1.8, "h_m": 1.8, "size_str": "1.8m × 1.8m", "image_path": "images/image-removebg-preview (26).png", "style_category": "Luxury"},
                {"name": "Ergonomic Accent Armchair", "cost": 450, "style": 20, "w_m": 0.85, "h_m": 0.85, "size_str": "0.85m × 0.85m", "image_path": "images/image-removebg-preview (5).png", "style_category": "Minimalist"},
                {"name": "Bamboo Slat Accent Rug", "cost": 250, "style": 15, "w_m": 1.2, "h_m": 1.6, "size_str": "1.2m × 1.6m", "image_path": "images/image-removebg-preview (3).png", "style_category": "Boho"},
                {"name": "Monochrome Polka Area Mat", "cost": 180, "style": 12, "w_m": 0.8, "h_m": 1.4, "size_str": "0.8m × 1.4m", "image_path": "images/image-removebg-preview (6).png", "style_category": "Kitsch"},
                {"name": "Crimson Lattice Runner", "cost": 210, "style": 18, "w_m": 0.9, "h_m": 1.8, "size_str": "0.9m × 1.8m", "image_path": "images/image-removebg-preview (7).png", "style_category": "Kitsch"},
                {"name": "Cream Waves Accent Rug", "cost": 220, "style": 15, "w_m": 1.1, "h_m": 1.6, "size_str": "1.1m × 1.6m", "image_path": "images/image-removebg-preview (10).png", "style_category": "Boho"},
                {"name": "Oval Marble Textured Mat", "cost": 195, "style": 14, "w_m": 1.3, "h_m": 0.9, "size_str": "1.3m × 0.9m", "image_path": "images/image-removebg-preview (11).png", "style_category": "Luxury"},
                {"name": "Low-Profile Floor Planter Rows", "cost": 130, "style": 10, "w_m": 1.4, "h_m": 0.4, "size_str": "1.4m × 0.4m", "image_path": "images/image-removebg-preview (4).png", "style_category": "Minimalist"},
                {"name": "Modern Sectional Sofa", "cost": 1850, "style": 60, "w_m": 2.6, "h_m": 1.8, "size_str": "2.6m × 1.8m", "image_path": "images/image-removebg-preview (12).png", "style_category": "Luxury"},
                {"name": "Classic Dining Set", "cost": 950, "style": 30, "w_m": 2.0, "h_m": 0.85, "size_str": "2.0m × 0.85m", "image_path": "images/image-removebg-preview (14).png", "style_category": "Boho"},
                {"name": "Sleek Loveseat Sofa", "cost": 750, "style": 25, "w_m": 1.6, "h_m": 0.85, "size_str": "1.6m × 0.85m", "image_path": "images/image-removebg-preview (15).png", "style_category": "Minimalist"},
                {"name": "Round Dining Set", "cost": 550, "style": 22, "w_m": 1.8, "h_m": 0.5, "size_str": "1.8m × 0.5m", "image_path": "images/image-removebg-preview (18).png", "style_category": "Luxury"},
                {"name": "Spacious Modular Wardrobe", "cost": 890, "style": 28, "w_m": 1.5, "h_m": 0.6, "size_str": "1.5m × 0.6m", "image_path": "images/image-removebg-preview (19).png", "style_category": "Luxury"},
                {"name": "Colorful Area Rug", "cost": 380, "style": 18, "w_m": 1.4, "h_m": 0.7, "size_str": "1.4m × 0.7m", "image_path": "images/image-removebg-preview (24).png", "style_category": "Kitsch"}
            ],
            "Bedroom": [
                {"name": "Scandinavian Double Bed Frame", "cost": 1650, "style": 55, "w_m": 2.1, "h_m": 2.3, "size_str": "2.1m × 2.3m", "image_path": "images/image-removebg-preview (9).png", "style_category": "Minimalist"},
                {"name": "Minimalist Single Platform Bed", "cost": 950, "style": 35, "w_m": 1.3, "h_m": 2.1, "size_str": "1.3m × 2.1m", "image_path": "images/image-removebg-preview (8).png", "style_category": "Minimalist"},
                {"name": "Mid-Century Modern Bed", "cost": 1100, "style": 40, "w_m": 2.1, "h_m": 0.9, "size_str": "2.1m × 0.9m", "image_path": "images/image-removebg-preview (13).png", "style_category": "Minimalist"},
                {"name": "Colorful Area Rug", "cost": 380, "style": 18, "w_m": 1.4, "h_m": 0.7, "size_str": "1.4m × 0.7m", "image_path": "images/image-removebg-preview (24).png", "style_category": "Kitsch"},
                {"name": "Spacious Modular Wardrobe", "cost": 890, "style": 28, "w_m": 1.5, "h_m": 0.6, "size_str": "1.5m × 0.6m", "image_path": "images/image-removebg-preview (19).png", "style_category": "Luxury"},
                {"name": "Low-Profile Floor Planter Rows", "cost": 130, "style": 10, "w_m": 1.4, "h_m": 0.4, "size_str": "1.4m × 0.4m", "image_path": "images/image-removebg-preview (4).png", "style_category": "Minimalist"},
                {"name": "Oval Marble Textured Mat", "cost": 195, "style": 14, "w_m": 1.3, "h_m": 0.9, "size_str": "1.3m × 0.9m", "image_path": "images/image-removebg-preview (11).png", "style_category": "Luxury"},
                {"name": "Monochrome Polka Area Mat", "cost": 180, "style": 12, "w_m": 0.8, "h_m": 1.4, "size_str": "0.8m × 1.4m", "image_path": "images/image-removebg-preview (6).png", "style_category": "Kitsch"},

            ]
        }

    class ProcessUtilitySuggestions(CyclicBehaviour):
        async def run(self):
            state = st.session_state.room_data
            
            active_style_group = state.get("user_style_preference", "Minimalist")
            active_type = state["room_type"]
            
            suggestions = []
            room_catalog = self.agent.catalog.get(active_type, [])
            
            for item in room_catalog:
                item_cat = item.get("style_category", "Minimalist")
                is_match = (item_cat == active_style_group)
                
                utility_score = 30
                if is_match: 
                    utility_score += 50 
                if item["cost"] <= state.get("budget", 5000): 
                    utility_score += 20
                
                if utility_score >= 60:
                    suggested_item = item.copy()
                    suggested_item["utility_score"] = utility_score
                    suggestions.append(suggested_item)
            
            st.session_state.room_data["designer_suggestions"] = suggestions

            locked_to = st.session_state.room_data.get("catalog_locked_to")
            if locked_to:
                log_agent_action("Designer Agent", f" Catalog locked to '{locked_to}' style (2+ items chosen).")
            else:
                log_agent_action("Designer Agent", f"Catalog updated for '{active_style_group}' items.")
            await asyncio.sleep(2.5)
            
    class ReceiveStyleFromManager(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg and msg.metadata.get("performative") == "inform":
                if msg.body and msg.body.startswith("NEW_STYLE|"):
                    new_style = msg.body.split("|", 1)[1]
                    st.session_state.room_data["user_style_preference"] = new_style
                    log_agent_action("Designer Agent", f"Style preference updated to '{new_style}' via Manager Agent signal.")
            await asyncio.sleep(0.5)

    class RespondToBudgetCooperation(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=5)
            if msg and msg.metadata.get("performative") == "request":
                try:
                    max_allowable_cost = float(msg.body)
                except ValueError:
                    max_allowable_cost = 1000.0
                
                state = st.session_state.room_data
                active_type = state["room_type"]
                
                alternatives = []
                for item in self.agent.catalog.get(active_type, []):
                    if item["cost"] <= max_allowable_cost:
                        alternatives.append(item)
                
                reply = msg.make_reply()
                reply.set_metadata("performative", "inform")
                if alternatives:
                    alternatives.sort(key=lambda x: x["style"], reverse=True)
                    best_option = alternatives[0]
                    reply.body = f"{best_option['name']}|{best_option['cost']}"
                else:
                    reply.body = "NONE_AVAILABLE"
                
                await self.send(reply)
            await asyncio.sleep(0.5)

    async def setup(self):
        self.add_behaviour(self.ProcessUtilitySuggestions())
        self.add_behaviour(self.ReceiveStyleFromManager())
        self.add_behaviour(self.RespondToBudgetCooperation())


class BudgetAgent(Agent):
    class EnforceGoalParameters(CyclicBehaviour):
        async def run(self):
            state = st.session_state.room_data
            total_spend = state["floor_cost"] + state["furniture_cost"]
            remaining = state["base_budget"] - total_spend
            st.session_state.room_data["budget"] = remaining
            
            if remaining < 0:
                alert = f" **Budget Agent Warning**: Deficit of ${abs(remaining):.2f}!"
                if alert not in st.session_state.room_data["alerts"]:
                    st.session_state.room_data["alerts"].append(alert)
                
                allowed_ceiling = max(0.0, state["furniture_cost"] + remaining)
                
                req = Message(to="designer_sub@localhost")
                req.set_metadata("performative", "request")
                req.body = str(allowed_ceiling)
                
                log_agent_action("Budget Agent", f"Sending FIPA request to Designer for items under ${allowed_ceiling:.2f}")
                await self.send(req)
                
                reply = await self.receive(timeout=3)
                if reply and reply.metadata.get("performative") == "inform":
                    if reply.body != "NONE_AVAILABLE":
                        name, cost = reply.body.split("|")
                        st.session_state.room_data["alerts"].append(
                            f" **Budget & Designer Cooperation**: Swap out items for `{name}` (${cost}) to preserve financial goals!"
                        )
            
            await asyncio.sleep(2.0)

    async def setup(self):
        self.add_behaviour(self.EnforceGoalParameters())

class ManagerAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_reported_style = "Minimalist"

    class ObserveUserPipeline(CyclicBehaviour):
        async def run(self):
            try:
                state = st.session_state.room_data
                current_style = state.get("user_style_preference", "Minimalist")

                if self.agent.last_reported_style != current_style:
                    self.agent.last_reported_style = current_style
                    log_agent_action("Manager Agent", f" Preference shift detected! Notifying Designer: '{current_style}'")

                    msg = Message(to="designer_sub@localhost")
                    msg.set_metadata("performative", "inform")
                    msg.body = f"NEW_STYLE|{current_style}"
                    await self.send(msg)

            except Exception as e:
                pass

            await asyncio.sleep(1.0)

    async def setup(self):
        self.add_behaviour(self.ObserveUserPipeline())



def execute_async_agent_thread(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

if "agents_started" not in st.session_state:
    bg_loop = asyncio.new_event_loop()
    worker_thread = threading.Thread(target=execute_async_agent_thread, args=(bg_loop,), daemon=True)
    worker_thread.start()

    arch_a = ArchitectAgent("architect_sub@localhost", "pass")
    struct_a = StructuralAgent("structural_sub@localhost", "pass")
    design_a = DesignerAgent("designer_sub@localhost", "pass")
    budget_a = BudgetAgent("budget_sub@localhost", "pass")
    manager_a = ManagerAgent("manager_sub@localhost", "pass")
    
    st.session_state.designer_catalog = design_a.catalog

    asyncio.run_coroutine_threadsafe(arch_a.start(), bg_loop)
    asyncio.run_coroutine_threadsafe(struct_a.start(), bg_loop)
    asyncio.run_coroutine_threadsafe(design_a.start(), bg_loop)
    asyncio.run_coroutine_threadsafe(budget_a.start(), bg_loop)
    asyncio.run_coroutine_threadsafe(manager_a.start(), bg_loop)

    st.session_state.agents_started = True

def trim_and_normalize_furniture(image_path, target_width_m, target_height_m, pxl_per_meter=200):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            bbox = img.getbbox()
            if bbox: img = img.crop(bbox)
        dest_w_px = max(int(target_width_m * pxl_per_meter), 1)
        dest_h_px = max(int(target_height_m * pxl_per_meter), 1)
        return img.resize((dest_w_px, dest_h_px), Image.Resampling.LANCZOS)
    except Exception:
        return Image.new("RGBA", (100, 100), (220, 220, 220, 160))

def update_immediate_metrics():
    state = st.session_state.room_data
    state["floor_cost"] = PRICING.get(state["flooring"], 0)
    
    total_furniture_cost = 0.0
    total_style_points = 0.0

    tally = {"Luxury": 0, "Boho": 0, "Minimalist": 0, "Kitsch": 0}
    for item in state.get("furniture", []):
        total_furniture_cost += item.get("cost", 0.0)
        total_style_points += item.get("style", 0.0)
        cat = item.get("style_category", "Minimalist")
        if cat in tally:
            tally[cat] += 1

    if state.get("furniture"):
        state["user_style_preference"] = max(tally, key=tally.get)
    else:
        state["user_style_preference"] = "Minimalist"

    dominant_style = state["user_style_preference"]
    dominant_count = tally.get(dominant_style, 0)
    full_catalog = st.session_state.get("designer_catalog_full", {})

    if not full_catalog:
        full_catalog = dict(st.session_state.get("designer_catalog", {}))
        st.session_state.designer_catalog_full = full_catalog

    if dominant_count >= 2:
        locked = {}
        for room_type, items in full_catalog.items():
            locked[room_type] = [i for i in items if i.get("style_category") == dominant_style]
        st.session_state.designer_catalog = locked
        state["catalog_locked_to"] = dominant_style
    else:
        st.session_state.designer_catalog = full_catalog
        state.pop("catalog_locked_to", None)

    state["furniture_cost"] = total_furniture_cost
    state["budget"] = state["base_budget"] - (state["floor_cost"] + state["furniture_cost"])
    state["style_score"] = total_style_points

def draw_room():
    w = st.session_state.room_data["width"]
    h = st.session_state.room_data["height"]
    floor = st.session_state.room_data["flooring"]
    w_color = st.session_state.room_data["wall_color"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color=w_color, lw=10, zorder=1)
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="black", lw=1, zorder=2)
    
    colors = {"Marble": "#E0E0E0", "Hardwood": "#C19A6B", "Carpet": "#B0C4DE", "None": "white"}
    ax.add_patch(plt.Rectangle((0, 0), w, h, color=colors.get(floor, "white"), alpha=0.4))
    
    for op in st.session_state.room_data["openings"]:
        color = "red" if op["type"] == "door" else "cyan"
        wall, pos, size = op["wall"], op["pos"], op["size"]
        h_size = size / 2
        if wall == 0: ax.plot([pos - h_size, pos + h_size], [0, 0], color=color, lw=12, solid_capstyle='butt', zorder=5)
        elif wall == 1: ax.plot([w, w], [pos - h_size, pos + h_size], color=color, lw=12, solid_capstyle='butt', zorder=5)
        elif wall == 2: ax.plot([pos - h_size, pos + h_size], [h, h], color=color, lw=12, solid_capstyle='butt', zorder=5)
        elif wall == 3: ax.plot([0, 0], [pos - h_size, pos + h_size], color=color, lw=12, solid_capstyle='butt', zorder=5)
            
    for item in st.session_state.room_data["furniture"]:
        img_path = Path(item["image_path"])
        img = trim_and_normalize_furniture(img_path, item["w_m"], item["h_m"])
        x_min, x_max = -item["w_m"] / 2, item["w_m"] / 2
        y_min, y_max = -item["h_m"] / 2, item["h_m"] / 2
        
        im = ax.imshow(img, extent=[x_min, x_max, y_min, y_max], zorder=6)
        trans = mtransforms.Affine2D().rotate_deg(item.get("angle", 0)).translate(item["x"], item["y"])
        im.set_transform(trans + ax.transData)

    ax.set_xlim(-1, w + 1); ax.set_ylim(-1, h + 1); ax.set_aspect('equal'); ax.axis('off')
    return fig

st.set_page_config(page_title="Bedroom/Livingroom decor manager", layout="wide")
st.title("Bedroom/Livingroom decor manager")

st.session_state.room_data["alerts"] = []
st.sidebar.header(" Layout Configuration")

with st.sidebar.expander("Step 1: Establish Budget", expanded=(st.session_state.room_data["base_budget"] == 5000.0)):
    custom_budget = st.number_input("Total Allocation ($)", min_value=500, max_value=50000, value=int(st.session_state.room_data["base_budget"]), step=500)
    if custom_budget != st.session_state.room_data["base_budget"]:
        st.session_state.room_data["base_budget"] = float(custom_budget)
        st.session_state.room_data["budget"] = float(custom_budget)

        log_agent_action("User Interaction", f"Adjusted global project budget allocation to ${custom_budget}")
        st.rerun()

with st.sidebar.expander(" Step 2: Select Room", expanded=False):
    room_types = ["Living Room", "Bedroom"]
    selected_room_type = st.selectbox("Active Profile", room_types, index=room_types.index(st.session_state.room_data["room_type"]))
    if selected_room_type != st.session_state.room_data["room_type"]:
        st.session_state.room_data["room_type"] = selected_room_type
        st.session_state.room_data["furniture"] = [] 
        log_agent_action("User Interaction", f"Switched active environment map profile to {selected_room_type}.")
        st.rerun()

with st.sidebar.expander("Step 3: Define Room Dimensions", expanded=False):
    new_w = st.number_input("Width (m)", 2.0, 15.0, st.session_state.room_data["width"], step=0.5)
    new_h = st.number_input("Height (m)", 2.0, 15.0, st.session_state.room_data["height"], step=0.5)
    if st.button("Commit Dimensions"):
        st.session_state.room_data["width"] = new_w
        st.session_state.room_data["height"] = new_h
        st.session_state.room_data["openings"] = [] 
        st.session_state.room_data["furniture"] = []
        log_agent_action("User Interaction", f"Altered shell grid parameters to {new_w}m x {new_h}m.")
        st.rerun()

with st.sidebar.expander(" Step 4: Architectural Management", expanded=False):
    st.markdown("#### Add New Opening")
    type_op = st.selectbox("Type", ["door", "window"])
    wall_side = st.selectbox("Wall Anchor", ["Bottom (0)", "Right (1)", "Top (2)", "Left (3)"])
    w_idx = int(wall_side.split("(")[1].replace(")", ""))
    
    limit = st.session_state.room_data["width"] if w_idx in [0, 2] else st.session_state.room_data["height"]
    size_op = st.slider("Opening Width (m)", 0.5, 4.0, 1.0, step=0.1)
    pos_op = st.slider("Position Coordinate", float(size_op/2), float(limit - (size_op/2)), float(limit/2), step=0.1)
    
    if st.button("Save"):
        if check_overlap(w_idx, pos_op, size_op): 
            st.error("❌ Boundaries overlapping pre-existing structural elements!")
        else:
            st.session_state.room_data["openings"].append({"type": type_op, "wall": w_idx, "pos": pos_op, "size": size_op})
            log_agent_action("User Interaction", f"Added architectural opening: {type_op} on Wall {w_idx}.")
            st.rerun()
            
    if st.session_state.room_data["openings"]:
        st.markdown("---")
        st.markdown("#### Delete")
        for idx, op in enumerate(st.session_state.room_data["openings"]):
            col_lbl, col_btn = st.columns([3, 1])
            with col_lbl:
                st.caption(f"{idx+1}. {op['type'].upper()} on Wall {op['wall']} (Pos: {op['pos']}m)")
            with col_btn:
                if st.button("❌", key=f"del_op_{idx}"):
                    popped = st.session_state.room_data["openings"].pop(idx)
                    log_agent_action("User Interaction", f"Deleted opening: {popped['type']} on Wall {popped['wall']}")
                    st.rerun()

with st.sidebar.expander(" Step 5: Floor & Wall Color", expanded=False):
    selected_floor = st.selectbox("Floor", list(PRICING.keys()), index=list(PRICING.keys()).index(st.session_state.room_data["flooring"]))
    if selected_floor != st.session_state.room_data["flooring"]:
        st.session_state.room_data["flooring"] = selected_floor
        st.session_state.room_data["floor_cost"] = PRICING[selected_floor]
        log_agent_action("User Interaction", f"Altered floor material to {selected_floor}.")
        st.rerun()
        
    selected_wall = st.color_picker("Wall Color", st.session_state.room_data["wall_color"])
    if selected_wall != st.session_state.room_data["wall_color"]:
        st.session_state.room_data["wall_color"] = selected_wall
        log_agent_action("User Interaction", f"Altered wall color scheme profile to {selected_wall}.")
        st.rerun()

with st.sidebar.expander("Step 6: Furniture Inventory", expanded=True):
    if st.session_state.room_data["furniture"]:
        st.markdown("####  Transform Placed Elements")
        for idx, item in enumerate(st.session_state.room_data["furniture"]):
            st.markdown(f"**{item['name']}**")
            new_x = st.slider("X Position (m)", 0.0, float(st.session_state.room_data["width"]), float(item["x"]), key=f"x_{idx}")
            new_y = st.slider("Y Position (m)", 0.0, float(st.session_state.room_data["height"]), float(item["y"]), key=f"y_{idx}")
            new_angle = st.slider("Rotation Angle (°)", 0, 360, int(item.get("angle", 0)), step=15, key=f"ang_{idx}")
            
            if new_x != item["x"] or new_y != item["y"] or new_angle != item.get("angle", 0):
                is_colliding_wall = check_wall_collision(item, new_x, new_y, new_angle, st.session_state.room_data["width"], st.session_state.room_data["height"])
                colliding_item_name = check_item_to_item_collision(idx, new_x, new_y, new_angle)
                
                if is_colliding_wall:
                    st.error(f"Movement blocked! Hit boundary walls.")
                elif colliding_item_name:
                    st.error(f"Intersecting with `{colliding_item_name}`.")
                else:
                    st.session_state.room_data["furniture"][idx].update({"x": new_x, "y": new_y, "angle": new_angle})
                    st.rerun()
            st.markdown("---")
    else:
        st.caption("No furniture deployed yet. Add assets using the main selection grid on the right.")

update_immediate_metrics()
col_canvas, col_designer = st.columns([1.2, 1])

with col_canvas:
    st.subheader("Room Layout Visualization")
    st.pyplot(draw_room())
    
    st.subheader("Manager Agent History")
    st.caption("Active multi-agent and user pipeline logs:")
    st.code("\n".join(st.session_state.room_data["history"][-8:]))

with col_designer:
    st.subheader("Agent Console")
    m1, m2, m3 = st.columns(3) 
    m1.metric("Available Funds", f"${st.session_state.room_data['budget']}")
    m2.metric("Style Score", f"{st.session_state.room_data['style_score']} pts")
    
    current_inferred_style = st.session_state.room_data.get("user_style_preference", "Minimalist")
    m3.metric("Detected Aesthetic Style", current_inferred_style)
    
    recommended_items = st.session_state.room_data.get("designer_suggestions", [])
    if recommended_items:
        for rec_item in recommended_items[:3]:
            st.info(f" **Suggestion**: `{rec_item['name']}` fits your **{rec_item['style_category']}** style perfectly! (Cost: ${rec_item['cost']})")
    else:
        st.caption("No recommendations found matching current budget boundaries.")

    active_type = st.session_state.room_data["room_type"]
    locked_to = st.session_state.room_data.get("catalog_locked_to")
    if locked_to:
        st.markdown(f"###  Available Furniture Catalog ({active_type})")
        st.success(f" **Style locked to {locked_to}** — showing only matching items (2+ {locked_to} pieces chosen).")
    else:
        st.markdown(f"###  Available Furniture Catalog ({active_type})")
    
    designer_catalog = st.session_state.get("designer_catalog", {})
    
    filtered_catalog = []
    if isinstance(designer_catalog, dict) and active_type in designer_catalog:
        for item in designer_catalog[active_type]:
            if isinstance(item, dict) and "name" in item:
                filtered_catalog.append(item)
                    
    for item in filtered_catalog:
        is_added = any(f["name"] == item["name"] for f in st.session_state.room_data["furniture"])
        with st.container(border=True):
            c_text, c_img = st.columns([1.8, 1])
            with c_text:
                st.markdown(f"**{item['name']}**")
                
                size_str = item.get("size_str", f"{item.get('w_m')}m x {item.get('h_m')}m")
                st.markdown(f"Scale footprint: `{size_str}`")
                
                style_pts = item.get("style", 10)
                st.caption(f"Price: ${item['cost']} | style points: +{style_pts}({item['style_category']})")
                
                if is_added:
                    if st.button("Drop Item", key=f"rm_{item['name']}", type="secondary"):
                        st.session_state.room_data["furniture"] = [f for f in st.session_state.room_data["furniture"] if f["name"] != item["name"]]
                        log_agent_action("User Interaction", f"Removed asset {item['name']} from room.")
                        st.rerun()
                else:
                    can_buy = st.session_state.room_data["budget"] >= item["cost"]
                    
                    if st.button("Add", key=f"add_{item['name']}", disabled=not can_buy, type="primary"):
                        purchased = item.copy()
                        center_x = float(st.session_state.room_data["width"] / 2)
                        center_y = float(st.session_state.room_data["height"] / 2)
                        
                        purchased.update({
                            "x": center_x, 
                            "y": center_y, 
                            "angle": 0,
                            "style": style_pts,
                            "style_category": item.get("style_category", "Minimalist")
                        })
                        
                        st.session_state.room_data["furniture"].append(purchased)
                        log_agent_action("User Interaction", f"Deployed furniture {item['name']} to coordinate center.")
                        st.rerun()
            with c_img:
                try:
                    st.image(item["image_path"], width="stretch")
                except Exception:
                    st.caption("Furniture Image Missing")