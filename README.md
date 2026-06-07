# MAS-Project-Medrea-Flavia

My project is called "Bedroom/Livingroom decor manager" 
The agents:
1. Architect agent (Information agent/model-based):  Holds information about the room size, window/door placement;
2. Structural agent (reactive agent/simple reflex): Holds information about the flooring and walls;
3.  Designer agent (proactive agent/utility-based): it is like a furniture catalog, keeps score of user's style and can make suggestions;
4. Budget agent (cooperative agent/goal-based): makes sure every acquisition is within budget, cooperates with Designer agent and approves aquisition or suggests something else suitable to the budget;
5. Manager agent (interface agent/learning): bridge between user and designer, observes user behavior.

The application is built using a decoupled architecture: Streamlit handles the interactive frontend user interface, while SPADE (Smart Python Agent Development Environment) orchestrates the asynchronous, message-based backend agent logic via XMPP.

How to run:
After cloning the repository and installing all dependencies open terminal and type:
```bash
streamlit run main.py
