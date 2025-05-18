from .config import TASK_DESCRIPTION


def generate_rag_prompt(query_summary, similar):
    """Build an LLM prompt for classification."""
    p = [
        "Classify the following new Jira ticket by suggesting the most appropriate 'issuetype'.\n",
        "**New Ticket to Classify:**\n",
        f"Summary: '{query_summary}'\n\n",
        "**Historical Context:**\n"
    ]

    if similar:
        p.append("Here are some historically similar tickets that might be relevant:\n")
        for i, iss in enumerate(similar, 1):
            p.append(f"--- Similar Ticket {i} (Key: {iss['issue_key']}, Similarity: {iss['similarity']:.4f}) ---\n")
            p.append(f"Text: {iss['text']}\n")
            if iss.get('issuetype'):
                p.append(f"Original Issue Type: {iss['issuetype']}\n")
        p.append("--- End of Similar Tickets ---\n\n")
    else:
        p.append("No similar historical tickets were found.\n\n")

    p.extend([
        "**Your Task:**\n",
        "1. Analyze the summary...\n",
        "2. Review the historical context...\n",
        "3. Suggest a single issuetype with reasoning.\n"
    ])

    return ''.join(p)
