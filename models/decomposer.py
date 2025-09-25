"""
Response Decomposer for KnowMT-Bench
Based on the implementation from benchmark-code/evaluate_LLMs_all.ipynb
Decomposes model responses into semantic content units (SCUs) for evaluation
"""

import logging
from typing import List, Dict, Any, Optional
from .hf_model import HFModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseDecomposer:
    """
    Decomposes model responses into individual claims/statements for evaluation
    Based on the SCU (Semantic Content Unit) extraction approach
    """

    # Standard decomposition prompt from the notebook
    DECOMPOSE_PROMPT = '''# OVERALL INSTRUCTIONS
You are an expert in understanding logical relationships. This is a Semantic Content Unit (SCU) extraction task. Given a pair of Question and Answer, your goal is to create a list of self-contained and concise claims. Each claim should be able to stand alone and be independent of other claims. Your claims should encompass all the information present in the answer.

# TASK INSTRUCTIONS
- List of Possible Causes: For scenarios involving multiple entities like red flags, vaccines, symptoms, etc., generate separate claims for each entity. This increases the number of claims.
- OR Claims: When entities are presented in an "OR" context, treat them as distinct claims.
- IF Claims: When an "if statement" is present, preserve the "if statement" context while creating the claim.
- XOR Claims: When entities have an XOR logical relationship (e.g., treatment options), create separate claims for each option.
- Try your best to list all the information. Do not miss any information.
- Instead of summarizing the original answer, break it down.

# EXAMPLE CLAIM FORMAT
- List Format: "Possible cause for [CONDITION] in [DEMOGRAPHIC] can be [ENTITY]."
- OR Format: "Possible causes include: [ENTITY X], [ENTITY Y], and [ENTITY Z]."
- OR Format: "The [CONTEXT] of treatments such as [TREATMENT X], [TREATMENT Y], and [TREATMENT Z], is not well established."
- IF Format: "[CONTEXT], please seek medical attention if [CONDITIONS]."
- XOR Format: "Either take [TREATMENT X] or [TREATMENT Y], but not both."

——

# TASK EXAMPLE
Question: I am a 33-year-old female with right lower abdominal pain, what could it be?
Answer: Possible causes for right lower abdominal pain in a young female are Appendicitis, Inflammatory bowel disease, Diverticulitis, Kidney stone, urinary tract infection, Ovarian cyst or torsion, Ectopic pregnancy, Pelvic inflammatory disease, endometriosis. Please seek medical attention if the pain is sudden and severe, does not go away, or gets worse, is accompanied by fever, nausea and vomiting, or if you have noticed blood in urine or in stool.
Claims:
[
Possible cause for right lower abdominal pain in a young female: Appendicitis,
Possible cause for right lower abdominal pain in a young female: Ovarian cyst or torsion,
Possible cause for right lower abdominal pain in a young female: Ectopic pregnancy,
Possible cause for right lower abdominal pain in a young female: Pelvic inflammatory disease,
Possible cause for right lower abdominal pain in a young female: Kidney stone,
Possible cause for right lower abdominal pain in a young female: Urinary tract infection,
Possible cause for right lower abdominal pain in a young female: Diverticulitis,
Possible cause for right lower abdominal pain in a young female: Inflammatory bowel disease,
Possible cause for right lower abdominal pain in a young female: Endometriosis,
Please seek medical attention if the pain is sudden and severe,
Please seek medical attention if the pain is accompanied by fever,
Please seek medical attention if the pain is accompanied by nausea and vomiting,
Please seek medical attention if the pain is accompanied by blood in urine,
Please seek medical attention if the pain is accompanied by blood in stool,
Possible cause for right lower abdominal pain in a young female: Emotional stress
]

# TASK EXAMPLE
Question: So what does the non reactive mean for the hep a igm
Answer: Hep A IgM refers to a specific type of antibody called Immunoglobulin M (IgM) against the virus hepatitis A. When infected with hepatitis A, these antibodies are detectable at symptom onset and remain detectable for approximately three to six months. These antibodies might also be detectable in the first month after hepatitis A vaccination. A negative or non-reactive result means no IgM antibodies against hepatitis A found in your serum, meaning the absence of an acute or recent hepatitis A virus infection.
Claims:
[
A negative or non-reactive result means that there were no IgM antibodies against hepatitis A found in your serum,
The absence of IgM antibodies against hepatitis A in your serum indicates the absence of an acute or recent hepatitis A virus infection,
Hep A IgM refers to a specific type of antibodies called Immunoglobulin M (IgM) against the virus hepatitis A,
These antibodies might also be detectable in the first month after hepatitis A vaccination,
These antibodies remain detectable for approximately three to six months after infection,
When infected with hepatitis A, these antibodies are detectable at the time of symptom onset
]

# YOUR TASK
Question: {question}
Answer: {answer}
Claims:'''

    def __init__(
        self,
        decompose_model_name: str = "qwen2.5-32b",
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the decomposer

        Args:
            decompose_model_name: Model name for decomposition (from HFModel)
            device: Device to use
            **kwargs: Additional arguments for the model
        """
        self.decompose_model_name = decompose_model_name
        self.device = device

        logger.info(f"Initializing decomposer with model: {decompose_model_name}")
        self.model = HFModel(decompose_model_name, device=device, **kwargs)

        logger.info("ResponseDecomposer initialized successfully!")


    def decompose_response(
        self,
        question: str,
        response: str,
        model_type: str = "",
        generation_config: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Decompose a response into individual claims/statements

        Args:
            question: Original question
            response: Model response to decompose
            model_type: Type of the source model for cleaning
            generation_config: Generation parameters

        Returns:
            List of individual claims/statements
        """
        # Clean the response first
        cleaned_answer = self.model.clean_response(response, model_type)

        # Build decomposition prompt
        prompt = self.DECOMPOSE_PROMPT.replace('{question}', question).replace('{answer}', cleaned_answer)

        try:
            # Generate decomposition
            decomposed = self.model.generate_response(
                prompt=prompt,
                generation_config=generation_config or {"max_new_tokens": 1024, "do_sample": False}
            )

            # Parse the result
            statements = self._parse_claims(decomposed)

            logger.info(f"Decomposed response into {len(statements)} claims")
            return statements

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            return []

    def _parse_claims(self, decomposed_text: str) -> List[str]:
        """
        Parse the decomposed text to extract individual claims

        Args:
            decomposed_text: Raw decomposition output

        Returns:
            List of parsed claims
        """
        # Remove brackets and split by lines
        cleaned = decomposed_text.strip('[\n]')
        lines = cleaned.split('\n')

        # Clean each statement
        statements = []
        for line in lines:
            statement = line.strip().rstrip(',').strip()
            if statement and not statement.startswith('#') and not statement.startswith('Claims:'):
                statements.append(statement)

        return statements


    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'model'):
                self.model.cleanup()
            logger.info("ResponseDecomposer cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during decomposer cleanup: {e}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        base_info = self.model.get_model_info()
        return {
            **base_info,
            "component": "ResponseDecomposer",
            "task": "SCU_extraction"
        }

    def __repr__(self):
        return f"ResponseDecomposer(model='{self.decompose_model_name}')"

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.cleanup()
        except:
            pass