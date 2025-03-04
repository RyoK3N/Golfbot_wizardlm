import os
import warnings
import math
import json
import requests
from langchain.tools import Tool
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings('ignore')
os.environ["LANGCHAIN_TRACING_V2"] = "false"


class MasterAgent:
    def __init__(self, llm):
        self.llm = llm 
        self.tools = []
        self.agent = None
        self.agent_executor = None
        self.checks = self._check_ollama()  # ping ollama to check if it is running 
        self.prompt_template = None
        self._tool_names = []  # Track available tool names
        
    def _check_ollama(self):
        """Check if Ollama is running and available."""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                return True
            return False
        except Exception:
            return False
    
    def _is_golf_related(self, query):
        """Check if the query is related to golf using the LLM."""
        prompt = f"""Determine if the following user query is related to golf. 
The query is considered golf-related if it's about:
- Golf equipment (clubs, balls, etc.)
- Golf techniques, swings, or mechanics
- Golf physics or calculations related to golf shots
- Golf courses, tournaments or players
- Any statistical or mathematical question specifically mentioning golf

Only respond with "YES" if the query is clearly golf-related, or "NO" if it's unrelated to golf.

User query: "{query}"
Is this golf-related?"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip().upper()
            return "YES" in response_text
        except Exception as e:
            print(f"Error checking golf relevance: {e}")
            return True
    
    def _get_default_prompt_template(self):
        """Returns the default prompt template for the agent."""
        # Include the list of available tools in the prompt
        tool_names_list = ", ".join([f"'{name}'" for name in self._tool_names])
        
        custom_prompt = f"""You are a golf assistant. Your task is to answer queries by strictly calling the provided tools and outputting only a JSON object with exactly two keys: "tool_call" and "final_answer". Do not output any extra text.

You can ONLY use the following tools: {tool_names_list}. Do not make up or invent new tools.

For each query, perform the following steps:
1. Analyze the question and identify if one of your available tools can answer it.
2. Extract the relevant numeric values from the input.
3. Immediately call the appropriate tool with these values.
4. Use the tool's output to formulate a final answer.
5. Output ONLY a JSON object with:
   - "tool_call": A string describing the tool invoked with parameters.
   - "final_answer": A one-sentence final answer that includes the tool's output.

If the user asks a general question like "how can you help me?" or similar, use one of your tools to provide a relevant answer or explain what tools you have available and what questions you can answer. Always format your response as JSON.

Examples:
Query: "If a golf ball travels 150 meters in 3 seconds, what is its speed?"
Output should be: {{{{\"tool_call\": \"GolfSpeedCalculator(distance=150, time=3)\", \"final_answer\": \"The speed of the golf ball is 50 m/s.\"}}}}

Query: "If a golf club opens by 5 degrees but closes by 2 degrees, what is the final face angle?"
Output should be: {{{{\"tool_call\": \"GolfFaceAngleCalculator(open_angle=5, close_angle=2)\", \"final_answer\": \"The final face angle is 3 degrees.\"}}}}

Do not include any commentary, only output the JSON.

Input: {{input}}
Agent scratchpad: {{agent_scratchpad}}
"""
        return PromptTemplate.from_template(custom_prompt)
    
    def set_custom_prompt(self, prompt_template):
        """Set a custom prompt template for the agent."""
        self.prompt_template = prompt_template
        
    def add_tool(self, tool):
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)
        self._tool_names.append(tool.name)
        
    def add_tools(self, tools_list):
        """Add multiple tools to the agent's toolkit."""
        self.tools.extend(tools_list)
        for tool in tools_list:
            self._tool_names.append(tool.name)
        
    def add_default_golf_tools(self):
        """Add the default golf-related tools to the agent."""
        
        # Define tool functions
        def calculate_speed(distance: float, time: float) -> float:
            """Calculates the speed of a golf ball given distance (meters) and time (seconds)."""
            if time <= 0:
                return "Time must be greater than zero."
            return round(distance / time, 2)

        def calculate_face_angle(open_angle: float, close_angle: float) -> float:
            """Calculates the net face angle of a golf club given opening and closing angles."""
            return round(open_angle - close_angle, 2)

        def estimate_ball_trajectory(speed: float, angle: float) -> float:
            """Estimates the distance a golf ball travels using projectile motion."""
            g = 9.81  # m/s^2
            angle_rad = math.radians(angle)
            distance = (speed ** 2 * math.sin(2 * angle_rad)) / g
            return round(distance, 2)

        def calculate_swing_speed(force: float, mass: float) -> float:
            """Calculates the swing speed given impact force and club mass."""
            if mass <= 0:
                return "Mass must be greater than zero."
            return round(force / mass, 2)
            
        def help_guide(query: str) -> str:
            """Provides help information about available golf tools and what queries can be answered."""
            return """I can help with the following golf-related calculations:
            - Calculate golf ball speed (given distance and time)
            - Calculate club face angle (given opening and closing angles)
            - Estimate ball trajectory (given speed and launch angle)
            - Calculate swing speed (given force and club mass)
            - Search for golf-related information online (if search is enabled)
            
            Please ask a specific question about any of these topics."""

        # Create tools
        speed_tool = Tool(
            name="GolfSpeedCalculator",
            func=calculate_speed,
            description="Calculates golf ball speed given distance (meters) and time (seconds). Example: distance=150, time=3."
        )

        face_angle_tool = Tool(
            name="GolfFaceAngleCalculator",
            func=calculate_face_angle,
            description="Calculates net face angle given open_angle and close_angle in degrees. Example: open_angle=5, close_angle=2."
        )

        trajectory_tool = Tool(
            name="BallTrajectoryEstimator",
            func=estimate_ball_trajectory,
            description="Estimates the distance a golf ball travels given speed (m/s) and launch angle (degrees). Example: speed=40, angle=30."
        )

        swing_speed_tool = Tool(
            name="SwingSpeedCalculator",
            func=calculate_swing_speed,
            description="Calculates swing speed given impact force (N) and club mass (kg). Example: force=500, mass=0.5."
        )
        
        help_tool = Tool(
            name="GolfHelpGuide",
            func=help_guide,
            description="Provides help information about available golf tools and what queries the assistant can answer."
        )
        
        self.add_tools([speed_tool, face_angle_tool, trajectory_tool, swing_speed_tool, help_tool])
    
    def add_tavily_search(self, api_key=None, max_results=3):
        """Add Tavily search tool to the agent."""
        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key
        elif "TAVILY_API_KEY" not in os.environ:
            raise ValueError("Tavily API key must be provided either as an argument or as an environment variable.")
        
        search_tool = TavilySearchResults(
            name="GolfWebSearch",
            max_results=max_results, 
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),
            description="Search the internet for golf-related information. Use this for general questions about golf equipment, techniques, or history."
        )
        self.add_tool(search_tool)
    
    def initialize_agent(self):
        """Initialize the agent with the current tools and prompt template."""
        if not self.tools:
            raise ValueError("No tools have been added to the agent.")
        
        if not self.checks:
            raise RuntimeError("Ollama is not running or not accessible.")
        
        # Generate the prompt template if it hasn't been set manually
        if not self.prompt_template:
            self.prompt_template = self._get_default_prompt_template()
            
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt_template)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True, 
            return_intermediate_steps=True
        )
        return self.agent_executor
    
    def query(self, input_text):
        """Run a query through the agent and return the result."""
        if not self.agent_executor:
            self.initialize_agent()
        
        # First check if the query is golf-related
        if not self._is_golf_related(input_text):
            # Return a polite message explaining the agent's scope
            return {
                "parsed_output": {
                    "tool_call": "TopicFilter",
                    "final_answer": "I'm sorry, but I can only assist with golf-related questions. Please ask me about golf equipment, techniques, physics, calculations, or other golf topics."
                },
                "success": True
            }
            
        # If we get here, the query is golf-related, proceed as normal
        response = self.agent_executor.invoke({"input": input_text})
        
        # Attempt to parse the response as JSON
        try:
            # Look for JSON in the output string
            output = response["output"]
            # Find all content within curly brackets
            if "{" in output and "}" in output:
                start_idx = output.find("{")
                end_idx = output.rfind("}") + 1
                json_part = output[start_idx:end_idx]
                try:
                    parsed = json.loads(json_part)
                    if "tool_call" not in parsed or "final_answer" not in parsed:
                        raise ValueError("JSON missing required keys")
                    return {
                        "parsed_output": parsed,
                        "raw_response": response,
                        "success": True
                    }
                except Exception:
                    pass
                
            forced_json = {
                "tool_call": "Unknown",
                "final_answer": output.strip()
            }
            
            return {
                "parsed_output": forced_json,
                "raw_response": response,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "raw_response": response,
                "raw": response["output"] if "output" in response else str(response),
                "success": False
            }


def create_web_app(master_agent):
    """Creates and configures the Flask web application."""
    app = Flask(__name__, template_folder='.')
    
    @app.route('/')
    def index():
        return render_template('chatbot.html')
    
    @app.route('/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        user_message = data.get('message', '')
        
        try:
            result = master_agent.query(user_message)
            if result["success"]:
                return jsonify({
                    "success": True,
                    "result": result["parsed_output"]
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result["error"],
                    "raw": result["raw"] if "raw" in result else "Unknown error"
                })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            })
    
    return app


if __name__ == "__main__":
    print("Initializing the Golf Assistant Chatbot...")
    
    llm = ChatOllama(
        model="wizardlm2",
        format="json",  
    )
    
    agent = MasterAgent(llm)
    agent.add_default_golf_tools()
    
    try:
        agent.add_tavily_search()
        print("Search capability added successfully")
    except ValueError as e:
        print(f"Web search not available: {e}")
    
    try:
        agent.initialize_agent()
        print("Agent initialized successfully")
    except RuntimeError as e:
        print(f"Failed to initialize agent: {e}")
        print("Make sure Ollama is running at http://localhost:11434")
        exit(1)
    
    app = create_web_app(agent)
    print("\n Golfbot is ready!")
    print("Open your browser and go to http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)


