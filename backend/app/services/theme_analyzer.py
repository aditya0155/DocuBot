import logging
from typing import List, Dict, Any
import google.generativeai as genai
import json # For parsing potential JSON output from LLM
import re # For cleaning LLM response to extract JSON

from ..core.config import settings
from ..models import schemas

logger = logging.getLogger(__name__)

class ThemeAnalyzerService:
    def __init__(self):
        """
        Initializes the ThemeAnalyzerService.
        """
        if not settings.gemini_api_key or settings.gemini_api_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.error("Google AI client not configured (API key missing or placeholder). Theme analysis will be impacted.")
            self.chat_model = None
        else:
            try:
                # Ensure genai is configured before trying to use it.
                genai.configure(api_key=settings.gemini_api_key)
                self.chat_model = genai.GenerativeModel(settings.gemini_chat_model_id)
                logger.info(f"Gemini chat model '{settings.gemini_chat_model_id}' initialized for ThemeAnalyzerService.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini chat model '{settings.gemini_chat_model_id}': {e}")
                self.chat_model = None

    async def analyze_themes(self, answers: List[schemas.IndividualAnswer], query_text: str) -> List[schemas.Theme]:
        """
        Analyzes a list of individual answers to identify common themes using an LLM.

        Args:
            answers (List[schemas.IndividualAnswer]): List of answers extracted from documents.
            query_text (str): The original user query.

        Returns:
            List[schemas.Theme]: A list of identified themes with summaries and supporting doc IDs.
        """
        if not self.chat_model:
            logger.error("Chat model not initialized. Cannot analyze themes.")
            return []
        
        if not answers:
            logger.info("No individual answers provided for theme analysis.")
            return []

        # Prepare the context for the LLM
        context_for_llm = ""
        for i, ans in enumerate(answers):
            context_for_llm += f"Answer {i+1} (from Document ID: {ans.doc_id}):\n"
            context_for_llm += f"Extracted Answer: {ans.extracted_answer}\n"
            context_for_llm += f"Original Chunk Text: {ans.text_chunk[:300]}...\n" # Include some context
            context_for_llm += "---\n"
        
        prompt = f"""
        Original User Query: "{query_text}"

        Below are several answers extracted from different documents in response to the query.
        Your task is to identify common themes or topics present across these answers.
        For each distinct theme you identify:
        1. Provide a concise name for the theme (e.g., "Regulatory Non-Compliance", "Penalty Justification").
        2. Write a brief summary of the theme based on the provided answers.
        3. List the Document IDs that support or are relevant to this theme.

        Extracted Answers:
        ---
        {context_for_llm}
        ---

        Please format your output as a JSON list, where each item is an object with the following keys:
        "theme_name": (string) The name of the theme.
        "theme_summary": (string) A brief summary of the theme.
        "supporting_doc_ids": (list of strings) Document IDs that support this theme.

        Example JSON output:
        [
          {{
            "theme_name": "Theme Name One",
            "theme_summary": "Summary of theme one based on the answers.",
            "supporting_doc_ids": ["DOC001", "DOC003"]
          }},
          {{
            "theme_name": "Theme Name Two",
            "theme_summary": "Summary of theme two, highlighting different aspects.",
            "supporting_doc_ids": ["DOC002", "DOC004", "DOC005"]
          }}
        ]

        If no clear themes can be identified, return an empty JSON list [].
        Focus only on information present in the "Extracted Answers" and their context.
        """

        logger.debug(f"Prompt for theme analysis:\n{prompt[:1000]}...") # Log a snippet of the prompt

        try:
            # Configure for JSON output if the model supports it directly,
            # or instruct clearly in the prompt as done above.
            # For Gemini, you might use specific generation_config if available for JSON mode.
            # For now, relying on prompt instruction for JSON.
            response = await self.chat_model.generate_content_async(prompt)
            
            response_text = response.text if hasattr(response, 'text') else response.parts[0].text if response.parts else "[]"
            logger.debug(f"Raw LLM response for themes: {response_text}")

            # Attempt to parse the JSON output
            try:
                # Clean the response text to better extract JSON
                # LLMs sometimes add ```json ... ``` or other text around the JSON.
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    themes_data = json.loads(json_str)
                else:
                    logger.warning("No JSON list found in LLM response for themes. Assuming no themes.")
                    themes_data = []
                    
                parsed_themes = []
                if isinstance(themes_data, list):
                    for theme_item in themes_data:
                        if isinstance(theme_item, dict) and \
                           "theme_name" in theme_item and \
                           "theme_summary" in theme_item and \
                           "supporting_doc_ids" in theme_item and \
                           isinstance(theme_item["supporting_doc_ids"], list):
                            parsed_themes.append(schemas.Theme(**theme_item))
                        else:
                            logger.warning(f"Skipping malformed theme item from LLM: {theme_item}")
                else:
                    logger.error(f"LLM theme analysis did not return a list. Response: {themes_data}")

                logger.info(f"Successfully parsed {len(parsed_themes)} themes from LLM response.")
                return parsed_themes
            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse JSON from LLM theme analysis response: {json_e}. Response was: {response_text}")
                return [] # Return empty list if JSON parsing fails

        except Exception as e:
            logger.error(f"Error during theme analysis with Gemini: {e}", exc_info=True)
            return []


# Example Usage (for local testing)
if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=settings.log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger.info("--- ThemeAnalyzerService Example ---")

    # Mock individual answers
    mock_answers = [
        schemas.IndividualAnswer(doc_id="DOC001", extracted_answer="The fine was imposed under section 15.", citation_source="DOC001, P4, Pr2", text_chunk="Section 15 details penalties for non-compliance."),
        schemas.IndividualAnswer(doc_id="DOC002", extracted_answer="Tribunal observed delay in disclosure violated Clause 49.", citation_source="DOC002, P2, Pr1", text_chunk="Clause 49 of LODR requires timely disclosures."),
        schemas.IndividualAnswer(doc_id="DOC003", extracted_answer="Statutory frameworks justify the penalties.", citation_source="DOC003, P1", text_chunk="The penalties are in line with statutory requirements."),
        schemas.IndividualAnswer(doc_id="DOC004", extracted_answer="Non-compliance with SEBI Act was noted.", citation_source="DOC004, S2", text_chunk="The SEBI Act outlines strict compliance rules.")
    ]
    test_query = "What were the reasons for penalties and regulatory issues?"

    theme_analyzer = ThemeAnalyzerService()

    async def run_theme_analysis_example():
        if not theme_analyzer.chat_model:
            logger.error("Cannot run example: Gemini chat model not initialized in ThemeAnalyzerService. Check API key.")
            return

        logger.info(f"Analyzing themes for query: '{test_query}' with {len(mock_answers)} answers.")
        themes = await theme_analyzer.analyze_themes(mock_answers, test_query)
        
        if themes:
            logger.info(f"Identified {len(themes)} themes:")
            for i, theme in enumerate(themes):
                logger.info(f"  Theme {i+1}: {theme.theme_name}")
                logger.info(f"    Summary: {theme.theme_summary}")
                logger.info(f"    Supporting Docs: {theme.supporting_doc_ids}")
        else:
            logger.info("No themes were identified in the example.")

    asyncio.run(run_theme_analysis_example())
    logger.info("--- ThemeAnalyzerService Example Finished ---")