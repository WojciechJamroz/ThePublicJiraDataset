from .config import TASK_DESCRIPTION
from typing import List, Dict, Any

# List of Apache issuetypes and descriptions (extracted from jira_issuetype_information.json)
APACHE_ISSUETYPES = [
    ("Improvement", "An improvement or enhancement to an existing feature or task."),
    ("Task", "A task that needs to be done."),
    ("Sub-task", "The sub-task of the issue."),
    ("New Feature", "A new feature of the product, which has yet to be developed."),
    ("Bug", "A problem which impairs or prevents the functions of the product."),
    ("Epic", "Issue type for a big user story that needs to be broken down."),
    ("Test", "A new unit, integration or system test."),
    ("Wish", "General wishlist item."),
    ("New JIRA Project", "A request for a new JIRA project to be set up."),
    ("RTC", "An RTC request."),
    ("TCK Challenge", "Challenges made against the Sun Compatibility Test Suite."),
    ("Question", "A formal question. Initially added for the Legal JIRA."),
    ("Temp", ""),
    ("Brainstorming", "A place to record back and forth on notions not yet formed enough to make a 'New Feature' or 'Task'."),
    ("Umbrella", "An overarching type made of sub-tasks."),
    ("Story", "Issue type for a user story."),
    ("Technical task", "A technical task."),
    ("Dependency upgrade", "Upgrading a dependency to a newer version."),
    ("Suitable Name Search", "A search for a suitable name for an Apache product."),
    ("Documentation", "Documentation or Website."),
    ("Planned Work", "Assigned specifically to Contractors by the VP Infra or or other VP/ Board Member."),
    ("New Confluence Wiki", "A request for a new Confluence Wiki to be set up."),
    ("New Git Repo", ""),
    ("Github Integration", ""),
    ("New TLP ", ""),
    ("New TLP - Common Tasks", ""),
    ("SVN->GIT Migration", ""),
    ("Blog - New Blog Request", ""),
    ("Blogs - New Blog User Account Request", ""),
    ("Blogs - Access to Existing Blog", ""),
    ("New Bugzilla Project", ""),
    ("SVN->GIT Mirroring", ""),
    ("IT Help", "For general IT problems and questions. Created by JIRA Service Desk."),
    ("Access", "For new system accounts or passwords. Created by JIRA Service Desk."),
    ("Request", ""),
    ("Project", "Which project does this relate to?"),
    ("Proposal", ""),
    ("GitBox Request", ""),
    ("Dependency", "Issue is dependent on ..."),
    ("Requirement", ""),
    ("Comment", ""),
    ("Outage", "Pagerduty will use this to create tickets when an Incident occurs."),
    ("Office Hours", "Issues designed to be discussed during Office Hours meetings."),
    ("Pending Review", "Acknowledged but not planned work, or long range feature request in need of scoping and prioritization."),
    ("Board Vote", ""),
    ("Director Vote", ""),
    ("Technical Debt", "")
]


def generate_rag_prompt(query_summary: str, similar: List[Dict[str, Any]]) -> str:
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

    # Add possible categories for the LLM to choose from
    p.append("**Possible Issue Types (choose only from this list):**\n")
    for name, desc in APACHE_ISSUETYPES:
        if desc:
            p.append(f"- {name}: {desc}\n")
        else:
            p.append(f"- {name}\n")
    p.append("\n")

    p.extend([
        "**Your Task:**\n",
        "1. Analyze the summary and context.\n",
        "2. Review the possible issue types above.\n",
        "3. Suggest the single most appropriate issuetype (from the list) and provide a brief reasoning.\n",
        "\n**Output Format:**\n",
        "Issuetype: <chosen type>\nReasoning: <your reasoning>\n"
    ])

    return ''.join(p)
