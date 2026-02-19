from typing import List,Dict
from langchain.schema import Document 

class evaluator:
    def evaluate_retrieval(self,query:str,retrieved_docs:List[Document],expected_keywords:List[str]=None)->Dict:
        if not retrieved_docs:
            return{"num_retrieved":0,
                   "keyword_coverage":0.0,
                   "avg_chunk_length":0,
                   "assessment":"FAIL,no documents retrieved"}
        results={
            "num_retrieved":len(retrieved_docs),
            "avg_chunk_length":sum(len(d.page_content) for d in retrieved_docs)//len(retrieved_docs)
        }
        if expected_keywords:
            all_text=" ".join(d.page_content.lower()for d in retrieved_docs)
            found =sum(1 for kw in expected_keywords if kw.lower()in all_text)
            results['keyword_coverage'] = found / max(len(query_words), 1)
            results['kewords_found'] = found
            results['keywords_total']=len(expected_keywords)
        else:
            #check if query terms appear in results 
            query_words = [w.lower() for w in query.split() if len(w)>3]
            all_text = " ".join(d.page_content.lower() for d in retrieved_docs)
            found = sum(1 for w in query_words if w in all_text)
            results['keyword_coverage']=found/max(len(query_words))
        

        if results['keyword_coverage']>=0.7:
            results['assessment'] = "good,high keyword overlap"
        elif results['keyword_coverage']>=0.4:
            results["assessment"] = "MODERATE - Partial keyword overlap"
        else:
            results["assessment"] = "POOR - Low keyword overlap"

        return results
    
    def evaluate_answer(self,question: str,answer: str,source_docs: List[Document],) -> Dict:
        results = {
            "answer_length": len(answer),
            "has_content": len(answer.strip()) > 10,
        }

        # Check if answer mentions it can't find info
        cant_find_phrases = ["cannot find", "not found", "no information", "not mentioned"]
        results["admits_no_info"] = any(p in answer.lower() for p in cant_find_phrases)

        # do words from the answer appear in sources?
        if source_docs:
            source_text = " ".join(d.page_content.lower() for d in source_docs)
            answer_words = [w.lower() for w in answer.split() if len(w) > 4]

            if answer_words:
                grounded = sum(1 for w in answer_words if w in source_text)
                results["grounding_score"] = grounded / len(answer_words)
            else:
                results["grounding_score"] = 0.0
        else:
            results["grounding_score"] = 0.0

        # Overall assessment
        if results["grounding_score"] >= 0.5:
            results["assessment"] = "GOOD - Answer appears grounded in sources"
        elif results["grounding_score"] >= 0.3:
            results["assessment"] = "MODERATE - Partially grounded"
        else:
            results["assessment"] = "POOR - Answer may not be grounded in sources"

        return results

    def run_test_suite(self,qa_chain,test_cases: List[Dict]) -> List[Dict]:
        all_results = []

        for i, test in enumerate(test_cases):
            print(f"Running test {i+1}/{len(test_cases)}: {test['question'][:50]}...")

            result = qa_chain.ask(test["question"])

            # Evaluate
            answer_eval = self.evaluate_answer(
                test["question"],
                result["answer"],
                [Document(page_content=s["content"]) for s in result["sources"]],
            )

            all_results.append({
                "question": test["question"],
                "answer": result["answer"][:200] + "...",
                "num_sources": result["num_sources"],
                **answer_eval,
            })

        # Summary
        avg_grounding = sum(r.get("grounding_score", 0) for r in all_results) / len(all_results)
        print(f"\nEVALUATION SUMMARY:")
        print(f"   Tests run: {len(all_results)}")
        print(f"   Average grounding score: {avg_grounding:.2f}")

        return all_results 