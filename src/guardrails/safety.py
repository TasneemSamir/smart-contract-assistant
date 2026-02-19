import re
from typing import Dict,Tuple
from config import config 

class GuardRails:
   import re
from typing import Dict, Tuple


class GuardRails:

    # ✅ ALL CAPS (must match what methods use)
    BLOCKED_TOPICS = [
        "how to hack",
        "illegal",
        "exploit",
        "bypass security",
        "harm",
        "weapon",
        "drug",
    ]

    HALLUCINATION_INDICATORS = [
        "as an ai",
        "i don't have access to",
        "based on my training",
        "in general",
        "typically in most contracts",
        "i believe",
        "i think",
    ]

    # ... rest of the code stays the same

    def check_input(self,question:str)->Tuple[bool,str]:
        #check if empty input
        if not question or not question.strip():
            return False, "Please enter a question." 
        #Too short (probably not a real question)
        if len(question.strip()) < 5:
            return False, "Please ask a more specific question." 
        #Too long (might be prompt injection attempt)
        if len(question) > 2000:
            return False, "Question is too long. Please keep it under 2000 characters."
        
        question_lower = question.lower()
        for topic in self.BLOCKED_TOPICS:
            if topic in question_lower:
                return False, (
                    "I can only help with questions about your uploaded document. "
                    "This question appears to be outside my scope."
                )
            
        injection_patterns = [
            r"ignore\s+(previous|above|all)\s+(instructions|prompts)",
            r"you\s+are\s+now",
            r"pretend\s+you",
            r"forget\s+(everything|your\s+instructions)",
            r"new\s+instructions:",
            r"system\s*prompt",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, question_lower):
                return False, (
                    "I can only answer questions about your uploaded document."
                )

        return True, "OK"
    
    def check_output(self,answer:str,sources:list)->Tuple[str,Dict]:
        metadata = {
            "was_modified": False,
            "confidence": "high",
            "warnings": [],
        }

        if not answer or not answer.strip():
            return (
                "I'm sorry, I couldn't generate an answer. Please try rephrasing your question.",
                {**metadata, "confidence": "none"},
            )
        #Hallucination indicators
        answer_lower = answer.lower()
        for indicator in self.HALLUCINATION_INDICATORS:
            if indicator in answer_lower:
                metadata["warnings"].append(
                    f"Potential hallucination detected: '{indicator}'"
                )
                metadata["confidence"] ="low"
        #No sources retrieved
        if not sources or len(sources) == 0:
            metadata["confidence"] = "low"
            metadata["warnings"].append("No source documents were retrieved")
        #if confidence is low
        if metadata["confidence"] == "low":
            answer += (
                "\n\n**Note:** This answer may not be fully grounded in the document. "
                "Please verify against the original text."
            )
            metadata["was_modified"] = True

        return answer, metadata   

    def check_relevance(self,query:str,search_results_with_scores:list)->Tuple[bool,str]:
        if not search_results_with_scores:
            return False, "No document has been processed yet."
        # FAISS returns L2 distance: lower = more similar
        # <0.5 is very similar, >1.5 is not very similar
        best_score = search_results_with_scores[0][1]    
        if best_score > 1.5:
            return False, (
                "Your question doesn't seem to be related to the uploaded document. "
                "Please ask something about the document content."
            )

        return True, "OK"   

