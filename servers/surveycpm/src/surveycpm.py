import os
import re
import json
import copy
from pathlib import Path
from typing import Any, List, Dict

from jinja2 import Template

from fastmcp.prompts import PromptMessage
from ultrarag.server import UltraRAG_MCP_Server


app = UltraRAG_MCP_Server("surveycpm")


def load_prompt_template(template_path: str | Path) -> Template:
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    with open(template_path, "r", encoding="utf-8") as f:
        template_content = f.read()
    return Template(template_content)


# ==================== Survey Manager Prompts ====================
# These functions correspond to OneSurveyManager.prepare_messages() in buffer_manager_v3.py

@app.prompt(output="instruction_ls,survey_ls,cursor_ls,template->prompt_ls")
def surveycpm_search(
    instruction_ls: List[str],
    survey_ls: List[Dict[str, Any]],
    cursor_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    """
    Generate search prompts for survey sections. Handles both initial and subsequent searches.
    Corresponds to OneSurveyManager.prepare_messages() lines 335-346
    """
    template: Template = load_prompt_template(template)
    ret = []
    for instruction, survey, cursor in zip(instruction_ls, survey_ls, cursor_ls):
        # Handle both empty survey (initial) and existing survey
        if not survey or survey == {}:
            survey_str = "There is no survey."
        else:
            survey_str = _print_tasknote(survey, abbr=True)
        
        p = template.render(
            user_query=instruction,
            current_outline=survey_str,
            current_instruction=f"You need to update {cursor}"
        )
        ret.append(p)
    return ret


@app.prompt(output="instruction_ls,retrieved_info_ls,init_plan_template->prompt_ls")
def surveycpm_init_plan(
    instruction_ls: List[str],
    retrieved_info_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    """
    Generate prompts for creating initial survey outline.
    Corresponds to OneSurveyManager.prepare_messages() lines 348-351
    """
    template: Template = load_prompt_template(template)
    ret = []
    for instruction, retrieved_info in zip(instruction_ls, retrieved_info_ls):
        p = template.render(
            user_query=instruction,
            current_information=retrieved_info
        )
        ret.append(p)
    return ret


@app.prompt(output="instruction_ls,survey_ls,cursor_ls,retrieved_info_ls,write_template->prompt_ls")
def surveycpm_write(
    instruction_ls: List[str],
    survey_ls: List[Dict[str, Any]],
    cursor_ls: List[str],
    retrieved_info_ls: List[str],
    template: str | Path,
) -> List[PromptMessage]:
    """
    Generate prompts for writing survey content.
    Corresponds to OneSurveyManager.prepare_messages() lines 353-361
    """
    template: Template = load_prompt_template(template)
    ret = []
    for instruction, survey, cursor, retrieved_info in zip(
        instruction_ls, survey_ls, cursor_ls, retrieved_info_ls
    ):
        survey_str = _print_tasknote_hire(survey, last_detail=True)
        p = template.render(
            user_query=instruction,
            current_survey=survey_str,
            current_instruction=f"You need to update {cursor}",
            current_information=retrieved_info
        )
        ret.append(p)
    return ret


@app.prompt(output="instruction_ls,survey_ls,cursor_ls,no_extend_ls,extend_plan_template->prompt_ls")
def surveycpm_extend_plan(
    instruction_ls: List[str],
    survey_ls: List[Dict[str, Any]],
    cursor_ls: List[str],
    no_extend_ls: List[bool],
    template: str | Path,
    template_info: str | Path | None = None,
) -> List[PromptMessage]:
    """
    Generate prompts for extending survey outline.
    Corresponds to OneSurveyManager.prepare_messages() lines 370-389
    In no_extend mode (True): uses abbr=False, simpler template
    In full mode (False): uses last_detail=True, info template
    """
    template: Template = load_prompt_template(template)
    template_info_obj = load_prompt_template(template_info) if template_info else None
    
    ret = []
    for instruction, survey, cursor, no_extend in zip(
        instruction_ls, survey_ls, cursor_ls, no_extend_ls
    ):
        if no_extend:
            survey_str = _print_tasknote(survey, abbr=False)
            p = template.render(
                user_query=instruction,
                current_survey=survey_str
            )
        else:
            survey_str = _print_tasknote_hire(survey, last_detail=True)
            if template_info_obj:
                p = template_info_obj.render(
                    user_query=instruction,
                    current_survey=survey_str,
                    current_instruction=f"You should decide whether to extend the plan under the current position: {cursor}"
                )
            else:
                p = template.render(
                    user_query=instruction,
                    current_survey=survey_str
                )
        ret.append(p)
    return ret


# ==================== Helper Functions ====================

def _abbr_one_line(string, abbr=True, tokenizer=None):
    """Abbreviate content to one line."""
    if isinstance(string, dict):
        if "content" in string and string["content"]:
            return _abbr_one_line(string["content"], abbr=abbr, tokenizer=tokenizer)
        elif "plan" in string:
            return "[PLAN] " + string["plan"].replace("\n", " ").strip()
        else:
            return ""
    else:
        if not string:
            return ""
        else:
            if abbr and tokenizer:
                tokens = tokenizer(string, return_tensors="pt")
                if tokens.input_ids.size(1) > 150:
                    decoded_prefix = tokenizer.decode(tokens.input_ids[0][:100], skip_special_tokens=True)
                    decoded_suffix = tokenizer.decode(tokens.input_ids[0][-50:], skip_special_tokens=True)
                    decoded = decoded_prefix + " ... " + decoded_suffix
                    return "[OK] " + decoded.replace("\n", " ").strip()
                else:
                    return "[OK] " + string.replace("\n", " ").strip()
            else:
                return "[OK] " + string.replace("\n", " ").strip()


def _to_one_line(string):
    """Convert content to one line."""
    if isinstance(string, dict):
        if "content" in string:
            if not string["content"]:
                return ""
            return "[OK] " + string["content"].replace("\n", " ").strip() + _to_one_line(string["content"])
        elif "plan" in string:
            return "[PLAN] " + string["plan"].replace("\n", " ").strip()
        else:
            return ""
    if not string:
        return ""
    else:
        return string.replace("\n", " ")


def _check_progress_postion(current_survey):
    """Check the current progress position in the survey."""
    if current_survey == {}:
        return "outline"
    else:
        if "sections" in current_survey:
            for i, section in enumerate(current_survey["sections"]):
                if "content" not in section:
                    return f"section-{i+1}"
                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        if "content" not in subsection:
                            return f"section-{i+1}.{j+1}"
                        if "subsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsections"]):
                                if "content" not in subsubsection:
                                    return f"section-{i+1}.{j+1}.{k+1}"
    return None


def _check_progress_postion_last_detail(current_survey):
    """Check the last completed position with detail."""
    if current_survey == {}:
        return "outline"
    else:
        titles = ["outline"]
        if "sections" in current_survey:
            for i, section in enumerate(current_survey["sections"]):
                if "content" not in section:
                    return titles[-1]
                else:
                    titles.append(f"section-{i+1}")
                if "subsections" in section:
                    for j, subsection in enumerate(section["subsections"]):
                        if "content" not in subsection:
                            return titles[-1]
                        else:
                            titles.append(f"section-{i+1}.{j+1}")
                        if "subsections" in subsection:
                            for k, subsubsection in enumerate(subsection["subsections"]):
                                if "content" not in subsubsection:
                                    return titles[-1]
                                else:
                                    titles.append(f"section-{i+1}.{j+1}.{k+1}")
    return titles[-1]


def _print_tasknote(current_survey, abbr=True):
    """Print survey structure as a formatted string."""
    string = ""
    if current_survey == {}:
        return "There is no survey."
    
    # title
    try:
        content = _abbr_one_line(current_survey["title"], abbr=False)
        string += f"# Title: {content}\n"
    except:
        string += f"# Title: None\n"

    to_line_func = _abbr_one_line

    if "sections" in current_survey:
        for i, section in enumerate(current_survey["sections"]):
            title_key = "name" if "name" in section else "title"
            name, content = section[title_key], to_line_func(section, abbr)
            string += f"# Section-{i+1} [{name}]: {content}\n"

            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    name, content = subsection[title_key], to_line_func(subsection, abbr)
                    string += f"    ## Section-{i+1}.{j+1} [{name}]: {content}\n"

                    if "subsections" in subsection:
                        for k, subsubsection in enumerate(subsection["subsections"]):
                            name, content = subsubsection[title_key], to_line_func(subsubsection, abbr)
                            string += f"        ### Section-{i+1}.{j+1}.{k+1} [{name}]: {content}\n"
    
    return string


def _print_tasknote_hire(current_survey, last_detail=False):
    """Print survey structure with hierarchical detail."""
    string = ""
    if current_survey == {}:
        return "There is no survey."
    
    # title
    try:
        content = _abbr_one_line(current_survey["title"], abbr=False)
        string += f"# Title: {content}\n"
    except:
        string += f"# Title: None\n"

    # sections
    if last_detail:
        now_section = _check_progress_postion_last_detail(current_survey)
    else:
        now_section = _check_progress_postion(current_survey)
    
    now_hire = now_section.count(".")  # 0, 1, 2
    
    if "sections" in current_survey:
        for i, section in enumerate(current_survey["sections"]):
            title_key = "name" if "name" in section else "title"
            if now_hire == 0 or (now_section.startswith(f"section-{i+1}") and now_hire == 1):
                to_line_func = _to_one_line
            else:
                to_line_func = _abbr_one_line
            name, content = section[title_key], to_line_func(section)
            string += f"# Section-{i+1} [{name}]: {content}\n"

            if "subsections" in section:
                for j, subsection in enumerate(section["subsections"]):
                    if (now_section.startswith(f"section-{i+1}") and now_hire == 1) or \
                       (now_section.startswith(f"section-{i+1}.{j+1}") and now_hire == 2):
                        to_line_func = _to_one_line
                    else:
                        to_line_func = _abbr_one_line
                    
                    name, content = subsection[title_key], to_line_func(subsection)
                    string += f"    ## Section-{i+1}.{j+1} [{name}]: {content}\n"

                    if "subsections" in subsection:
                        for k, subsubsection in enumerate(subsection["subsections"]):
                            if now_section.startswith(f"section-{i+1}.{j+1}"):
                                to_line_func = _to_one_line
                            else:
                                to_line_func = _abbr_one_line
                            
                            name, content = subsubsection[title_key], to_line_func(subsubsection)
                            string += f"        ### Section-{i+1}.{j+1}.{k+1} [{name}]: {content}\n"
    
    return string


def _match_reference(text: str) -> List[str]:
    """
    Extract citation keys from LaTeX text.
    Corresponds to OneSurveyManager._match_reference() lines 545-570
    """
    reg = r"\\\w*cite(?!style)\w*\{(.+?)\}"
    placeholder_reg = re.compile(r"^#\d+$")
    reg_bibkeys = re.findall(reg, text)
    bibkeys = set()
    for bibkey in reg_bibkeys:
        single_bib = bibkey.split(",")
        for bib in single_bib:
            if not placeholder_reg.match(bib):
                bib = bib.strip()
                if bib and bib != "*":
                    bibkeys.add(bib)
                    
    reg = r"\\nocite{(.+?)\}"
    reg_bibkeys = re.findall(reg, text)
    for bibkey in reg_bibkeys:
        single_bib = bibkey.split(",")
        for bib in single_bib:
            if not placeholder_reg.match(bib):
                bib = bib.strip()
                if bib and bib != "*":
                    if bib in bibkeys:
                        bibkeys.remove(bib)
        
    ref_key_list = list(bibkeys)
    return ref_key_list


def _check_language_consistency(item: Any, user_instruction: str) -> bool:
    """
    Check if text language matches user instruction language.
    Corresponds to OneSurveyManager._check_language_consistency() lines 624-674
    """
    if isinstance(item, str):
        text = item
    elif isinstance(item, dict):
        text = ""
        for v in item.values():
            if isinstance(v, str):
                text += v + "\n"
            elif isinstance(v, list):
                for vv in v:
                    if isinstance(vv, str):
                        text += vv + "\n"
                    elif isinstance(vv, dict):
                        for vvv in vv.values():
                            if isinstance(vvv, str):
                                text += vvv + "\n"
    elif isinstance(item, list):
        text = ""
        for v in item:
            if isinstance(v, str):
                text += v + "\n"
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str):
                        text += vv + "\n"
                    elif isinstance(vv, list):
                        for vvv in vv:
                            if isinstance(vvv, str):
                                text += vvv + "\n"
    else:
        return False
    
    text = text.strip()
    text = text.replace(" ", "").replace("\n", "").replace("\t", "")
    text = re.sub(r'(\\\\cite)\{(.*?)\}', '', text, flags=re.DOTALL)
    text = re.sub(r'(\\cite)\{(.*?)\}', '', text, flags=re.DOTALL)
    # Remove punctuation
    comma_english = r'[!"#$%&\'()\*\+,-./:;<=>\?@\\\[\]^_`{\|}~]'
    text = re.sub(comma_english, "", text)
    if len(text) == 0:
        return True
    
    is_chinese = re.search(r'[\u4e00-\u9fff]', user_instruction) is not None
    
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_count = len(chinese_chars)
    total_chars = len(text)
    if is_chinese:
        return chinese_count / total_chars > 0.6  # 60% of the text is Chinese
    else:
        return chinese_count / total_chars < 0.3  # Less than 30% is Chinese


# ==================== Response Parsing Tools ====================
# These tools correspond to OneSurveyManager methods

@app.tool()
def parse_survey_response(
    response_text: str,
    is_json: bool = True
) -> Dict[str, Any]:
    """
    Parse LLM response for survey generation.
    Corresponds to OneSurveyManager._parse_response() lines 466-542
    
    Expected format:
    <thought> ... </thought>
    <action> {"name": "search", "arguments": {...}} </action>
    
    Returns:
        dict with keys: thought, action, parse_success, step_format
    """
    extracted_result = {}
    
    # patterns
    think_pattern = r"<thought>(.*?)</thought>"
    action_pattern = r"<action>(.*?)</action>"
    
    # extract information
    think_is_valid, action_is_valid = False, False
    
    think_match = re.search(think_pattern, response_text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
        think_is_valid = True
    else:
        think = ""
    extracted_result["thought"] = think
    
    if is_json:
        action_match = re.search(action_pattern, response_text, re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            try:
                action = json.loads(action)
                action_is_valid = True
            except:
                action_is_valid = False
                action = {}
        else:
            action_is_valid = False
            action = {}
    else:
        action_match = re.search(action_pattern, response_text, re.DOTALL | re.MULTILINE)
        if action_match:
            action = action_match.group(1).strip()
            action_is_valid = True
        else:
            action = ""
        action = {"name": "write", "content": action}
    
    extracted_result["action"] = action
    extracted_result["parse_success"] = action_is_valid
    
    score = 0.0
    if not think_is_valid:
        score -= 1.0
    if not action_is_valid:
        score -= 2.0
    
    extracted_result["step_format"] = {
        "score": score,
        "thought": think_is_valid,
        "action": action_is_valid,
    }
    
    return extracted_result


@app.tool()
def validate_survey_action(
    action: Dict[str, Any],
    valid_actions: List[str],
    current_survey: Dict[str, Any] | None = None,
    cursor: str | None = None,
    user_instruction: str | None = None,
    hard_mode: bool = False,
    retrieved_bibkeys: List[str] | None = None
) -> bool:
    """
    Validate if a survey action is properly formatted.
    Corresponds to OneSurveyManager._check_action_validity_no_state() lines 761-837
    (Stricter version with exact key validation)
    
    Args:
        action: The action dictionary to validate
        valid_actions: List of valid action names for current state
        current_survey: Current survey structure (optional)
        cursor: Current position in survey (optional)
        user_instruction: User's instruction (optional)
        hard_mode: Enable stricter validation (optional)
        retrieved_bibkeys: List of valid bibkeys from search (optional, for write validation)
    
    Returns:
        bool: True if action is valid
    """
    if not isinstance(action, dict):
        return False
    if "name" not in action:
        return False
    
    # check tool validity
    if action["name"] not in valid_actions:
        return False
    
    try:
        if action["name"] == "search":
            assert "keywords" in action
            assert isinstance(action["keywords"], list)
            assert len(action["keywords"]) > 0
            assert action.keys() == {"name", "keywords"}
            for kw in action["keywords"]:
                assert isinstance(kw, str) and len(kw) > 0
            if hard_mode:
                assert len(action["keywords"]) <= 5
                
        elif action["name"] == "init-plan":
            assert "title" in action
            assert "sections" in action
            assert isinstance(action["title"], str) and len(action["title"]) > 0
            assert isinstance(action["sections"], list) and len(action["sections"]) > 0
            assert action.keys() == {"name", "title", "sections"}
            for sec in action["sections"]:
                assert isinstance(sec, dict)
                assert "title" in sec and "plan" in sec
                assert isinstance(sec["title"], str) and len(sec["title"]) > 0
                assert isinstance(sec["plan"], str) and len(sec["plan"]) > 0
                assert sec.keys() == {"title", "plan"}
            if hard_mode:
                assert 3 <= len(action["sections"]) <= 12
                if user_instruction:
                    assert _check_language_consistency(
                        {"title": action["title"], "sections": action["sections"]}, 
                        user_instruction
                    )
                
        elif action["name"] == "extend-plan":
            assert "position" in action
            assert "subsections" in action
            assert isinstance(action["position"], str) and len(action["position"]) > 0
            assert isinstance(action["subsections"], list) and len(action["subsections"]) > 0
            assert action.keys() == {"name", "position", "subsections"}
            if cursor is not None:
                assert action["position"] == cursor
            assert action["position"].count(".") < 2
            for sec in action["subsections"]:
                assert isinstance(sec, dict)
                assert "title" in sec and "plan" in sec
                assert isinstance(sec["title"], str) and len(sec["title"]) > 0
                assert isinstance(sec["plan"], str) and len(sec["plan"]) > 0
                assert sec.keys() == {"title", "plan"}
            if hard_mode:
                assert 2 <= len(action["subsections"]) <= 7
                if user_instruction:
                    assert _check_language_consistency(
                        {"subsections": action["subsections"]}, 
                        user_instruction
                    )
                
        elif action["name"] == "nop":
            assert action.keys() == {"name"}
            
        elif action["name"] == "write":
            assert "content" in action
            assert action.keys() == {"name", "content"}
            if hard_mode:
                assert "#" not in action["content"]
                assert "bibkey" not in action["content"].lower()
                assert len(action["content"].strip()) > 100
                if user_instruction:
                    assert _check_language_consistency(action["content"], user_instruction)
                # Check citations are valid
                ref_key_list = _match_reference(action["content"])
                if retrieved_bibkeys:
                    for ref_key in ref_key_list:
                        if ref_key not in retrieved_bibkeys:
                            return False
                assert action["content"].count("\\cite") < 10
                
    except:
        return False
    
    return True


@app.tool()
def update_survey_position(
    survey: Dict[str, Any],
    position: str,
    update_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update survey content at a specific position.
    Corresponds to TaskNote_Manager.update_tasknote_stable() lines 209-247
    
    Args:
        survey: The survey structure
        position: Position to update (e.g., "section-1.2")
        update_data: Content to update with (e.g., {"content": "..."} or {"subsections": [...]})
    
    Returns:
        Updated survey structure (modified in place)
    """
    # Make a deep copy to avoid modifying the original
    survey = copy.deepcopy(survey)
    
    current = survey
    if position == "outline":
        # Update outline
        for key, value in update_data.items():
            current[key] = value
    else:
        parts = position.split('-')[1].split('.')
        indices = [int(part) - 1 for part in parts]
        for i, idx in enumerate(indices):
            if i == 0:
                current = current['sections'][idx]
            else:
                current = current['subsections'][idx]
        
        for key, value in update_data.items():
            current[key] = value
    
    return survey


@app.tool()
def get_survey_position(
    survey: Dict[str, Any],
    position: str,
    tag: str = "content"
) -> Any:
    """
    Get content at a specific position in the survey.
    Corresponds to TaskNote_Manager.get_tasknote_stable() lines 250-280
    
    Args:
        survey: The survey structure
        position: Position to get (e.g., "section-1.2")
        tag: What to get ("content" or "outline")
    
    Returns:
        Content at the specified position
    """
    parts = position.split('-')[1].split('.')
    indices = [int(part) - 1 for part in parts]
    current = survey
    
    for i, idx in enumerate(indices):
        if i == 0:
            current = current['sections'][idx]
        else:
            current = current['subsections'][idx]
    
    if tag == "outline":
        return current
    elif tag == "content":
        return current.get('content', "")
    else:
        raise ValueError(f"Invalid tag: {tag}")


# ==================== State Management Tools ====================
# These tools implement the state machine logic from OneSurveyManager


@app.tool()
def surveycpm_state_init(
    instruction_ls: List[str]
) -> Dict[str, List]:
    """
    Initialize survey state for all instances.
    Corresponds to OneSurveyManager.__init__() lines 292-322
    
    Returns initial state: search, cursor: outline, empty survey, etc.
    """
    n = len(instruction_ls)
    return {
        "state_ls": ["search"] * n,
        "cursor_ls": ["outline"] * n,
        "survey_ls": [{} for _ in range(n)],  # Create separate dict for each instance
        "retrieved_info_ls": [""] * n,
        "step_ls": [0] * n,
        "extend_time_ls": [0] * n,
        "no_check_ls": [True] * n,
        "no_extend_ls": [True] * n,
    }


@app.tool()
def surveycpm_state_router(
    state_ls: List[str]
) -> Dict[str, List[Dict[str, str]]]:
    """
    Route each instance to its appropriate state branch.
    """
    routed = [{"data": state, "state": state} for state in state_ls]
    return {"state_ls": routed}


@app.tool()
def surveycpm_parse_search_response(
    response_ls: List[str]
) -> Dict[str, List]:
    """
    Parse search responses and extract keywords.
    Corresponds to OneSurveyManager._parse_response() for search action
    """
    keywords_ls = []
    parse_success_ls = []
    
    for response in response_ls:
        result = parse_survey_response(
            response=response,
            is_json=True,
            valid_actions=["search"],
            hard_mode=False
        )
        
        keywords = result.get("action", {}).get("keywords", [])
        parse_success = result.get("parse_success", False) and len(keywords) > 0
        
        keywords_ls.append(keywords)
        parse_success_ls.append(parse_success)
    
    return {
        "keywords_ls": keywords_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool()
def surveycpm_after_search(
    ret_psg: List[Any],
    cursor_ls: List[str],
    parse_success_ls: List[bool]
) -> Dict[str, List]:
    """
    Process search results and determine next state.
    Corresponds to OneSurveyManager.state_manager() lines 897-905
    and interact_with_search_engine() lines 848-872
    
    If parse failed: state stays "search"
    If cursor="outline": next_state="analyst-init_plan"
    Else: next_state="write"
    """
    retrieved_info_ls = []
    state_ls = []
    
    for psg, cursor, parse_success in zip(ret_psg, cursor_ls, parse_success_ls):
        # Check if search succeeded (has results)
        search_success = psg is not None and len(psg) > 0
        
        if not parse_success or not search_success:
            # Parse or search failed: stay in search state
            state_ls.append("search")
            retrieved_info_ls.append("")
        else:
            # Extract information from search results
            if isinstance(psg, list):
                # Assuming psg is list of documents with 'summary' or 'text' field
                summaries = []
                for doc in psg:
                    if isinstance(doc, dict):
                        summaries.append(doc.get("summary", doc.get("text", "")))
                    elif isinstance(doc, str):
                        summaries.append(doc)
                info = "\n\n".join(summaries).strip()
            else:
                info = str(psg).strip()
            
            retrieved_info_ls.append(info if info else "")
            
            # Determine next state based on cursor
            if cursor == "outline":
                state_ls.append("analyst-init_plan")
            else:
                state_ls.append("write")
    
    return {
        "retrieved_info_ls": retrieved_info_ls,
        "state_ls": state_ls
    }


@app.tool()
def surveycpm_after_init_plan(
    response_ls: List[str],
    survey_ls: List[Dict[str, Any]]
) -> Dict[str, List]:
    """
    Parse init_plan responses and create survey structures.
    Corresponds to OneSurveyManager.state_manager() lines 907-916
    
    Creates survey with title and sections, moves to search state.
    """
    new_survey_ls = []
    state_ls = []
    cursor_ls = []
    parse_success_ls = []
    
    for response, survey in zip(response_ls, survey_ls):
        result = parse_survey_response(
            response=response,
            is_json=True,
            valid_actions=["init-plan"],
            hard_mode=False
        )
        
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        
        if parse_success and action.get("name") == "init-plan":
            # Create new survey
            new_survey = {
                "title": action.get("title", ""),
                "sections": action.get("sections", [])
            }
            new_survey_ls.append(new_survey)
            state_ls.append("search")
            # Update cursor to first section
            cursor_ls.append(_check_progress_postion(new_survey))
            parse_success_ls.append(True)
        else:
            # Parse failed: keep current state
            new_survey_ls.append(survey)
            state_ls.append("analyst-init_plan")
            cursor_ls.append("outline")
            parse_success_ls.append(False)
    
    return {
        "survey_ls": new_survey_ls,
        "state_ls": state_ls,
        "cursor_ls": cursor_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool()
def surveycpm_after_write(
    response_ls: List[str],
    survey_ls: List[Dict[str, Any]],
    cursor_ls: List[str],
    no_check_ls: List[bool],
    no_extend_ls: List[bool]
) -> Dict[str, List]:
    """
    Parse write responses and update survey content.
    Corresponds to OneSurveyManager.state_manager() lines 918-940
    
    In no_extend mode: always goes to search after successful write.
    """
    new_survey_ls = []
    state_ls = []
    new_cursor_ls = []
    parse_success_ls = []
    
    for response, survey, cursor, no_check, no_extend in zip(
        response_ls, survey_ls, cursor_ls, no_check_ls, no_extend_ls
    ):
        result = parse_survey_response(
            response=response,
            is_json=False,
            valid_actions=["write"],
            hard_mode=False
        )
        
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        
        if parse_success and action.get("name") == "write":
            content = action.get("content", "")
            if content:
                # Update survey content
                new_survey = update_survey_position(
                    survey=survey,
                    position=cursor,
                    update_data={"content": content}
                )
                new_survey_ls.append(new_survey)
                
                # In no_check and no_extend mode: always go to search
                if no_check and no_extend:
                    state_ls.append("search")
                    new_cursor_ls.append(_check_progress_postion(new_survey))
                else:
                    # Other modes not implemented for no_extend
                    state_ls.append("search")
                    new_cursor_ls.append(_check_progress_postion(new_survey))
                
                parse_success_ls.append(True)
            else:
                # No content: parse failed
                new_survey_ls.append(survey)
                state_ls.append("write")
                new_cursor_ls.append(cursor)
                parse_success_ls.append(False)
        else:
            # Parse failed: keep current state
            new_survey_ls.append(survey)
            state_ls.append("write")
            new_cursor_ls.append(cursor)
            parse_success_ls.append(False)
    
    return {
        "survey_ls": new_survey_ls,
        "state_ls": state_ls,
        "cursor_ls": new_cursor_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool()
def surveycpm_after_extend(
    response_ls: List[str],
    survey_ls: List[Dict[str, Any]],
    cursor_ls: List[str],
    extend_time_ls: List[int]
) -> Dict[str, List]:
    """
    Parse extend responses and handle extend-plan/nop actions.
    Corresponds to OneSurveyManager.state_manager() lines 956-975
    
    If extend-plan: add subsections, move to search
    If nop: set extend_time=12 to finish
    """
    new_survey_ls = []
    state_ls = []
    new_cursor_ls = []
    new_extend_time_ls = []
    parse_success_ls = []
    
    for response, survey, cursor, extend_time in zip(
        response_ls, survey_ls, cursor_ls, extend_time_ls
    ):
        result = parse_survey_response(
            response=response,
            is_json=True,
            valid_actions=["extend-plan", "nop"],
            hard_mode=False
        )
        
        parse_success = result.get("parse_success", False)
        action = result.get("action", {})
        action_name = action.get("name", "")
        
        if parse_success and action_name == "extend-plan":
            position = action.get("position", "")
            subsections = action.get("subsections", [])
            
            if position and subsections:
                # Add subsections to survey (subsections are deep copied in update_survey_position)
                new_survey = update_survey_position(
                    survey=survey,
                    position=position,
                    update_data={"subsections": copy.deepcopy(subsections)}
                )
                new_survey_ls.append(new_survey)
                state_ls.append("search")
                new_cursor_ls.append(_check_progress_postion(new_survey))
                new_extend_time_ls.append(extend_time)
                parse_success_ls.append(True)
            else:
                # Invalid action
                new_survey_ls.append(survey)
                state_ls.append("analyst-extend_plan")
                new_cursor_ls.append(cursor)
                new_extend_time_ls.append(extend_time)
                parse_success_ls.append(False)
                
        elif parse_success and action_name == "nop":
            # No operation: finish extension
            new_survey_ls.append(survey)
            state_ls.append("search")
            new_cursor_ls.append(cursor)
            new_extend_time_ls.append(12)  # Set to max to prevent further extension
            parse_success_ls.append(True)
        else:
            # Parse failed: keep current state
            new_survey_ls.append(survey)
            state_ls.append("analyst-extend_plan")
            new_cursor_ls.append(cursor)
            new_extend_time_ls.append(extend_time)
            parse_success_ls.append(False)
    
    return {
        "survey_ls": new_survey_ls,
        "state_ls": state_ls,
        "cursor_ls": new_cursor_ls,
        "extend_time_ls": new_extend_time_ls,
        "parse_success_ls": parse_success_ls
    }


@app.tool()
def surveycpm_increment_step(
    step_ls: List[int]
) -> Dict[str, List[int]]:
    """
    Increment step counter for all instances.
    """
    return {"step_ls": [step + 1 for step in step_ls]}


@app.tool()
def surveycpm_check_completion(
    cursor_ls: List[str | None],
    extend_time_ls: List[int],
    no_extend_ls: List[bool],
    state_ls: List[str],
    step_ls: List[int],
    max_step: int = 140
) -> Dict[str, List]:
    """
    Check if survey generation is complete.
    Corresponds to OneSurveyManager.state_manager() lines 986-999
    
    When cursor is None (all sections done):
      - If extend_time < 12 and no_extend=True: go to analyst-extend_plan
      - Else: done
    If step >= max_step: done (incomplete)
    """
    new_state_ls = []
    new_extend_time_ls = []
    done_ls = []
    
    for cursor, extend_time, no_extend, state, step in zip(
        cursor_ls, extend_time_ls, no_extend_ls, state_ls, step_ls
    ):
        # Check if max step reached
        if step >= max_step:
            new_state_ls.append("done")
            new_extend_time_ls.append(extend_time)
            done_ls.append(False)  # Incomplete
            continue
        
        # Check if all sections are complete (cursor is None)
        if cursor is None:
            if extend_time < 12 and no_extend:
                # Still can extend
                new_state_ls.append("analyst-extend_plan")
                new_extend_time_ls.append(extend_time + 1)
                done_ls.append(False)
            else:
                # Finished
                new_state_ls.append("done")
                new_extend_time_ls.append(extend_time)
                done_ls.append(True)
        else:
            # Not yet complete: keep current state
            new_state_ls.append(state)
            new_extend_time_ls.append(extend_time)
            done_ls.append(False)
    
    return {
        "state_ls": new_state_ls,
        "extend_time_ls": new_extend_time_ls,
        "done_ls": done_ls
    }


@app.tool()
def surveycpm_check_all_done(
    done_ls: List[bool]
) -> Dict[str, List[Dict[str, str]]]:
    """
    Check if all surveys are done.
    """
    if all(done_ls):
        state = "all_done"
    else:
        state = "not_done"
    
    return {"done_ls": [{"data": str(done), "state": state} for done in done_ls]}


@app.tool()
def surveycpm_format_output(
    survey_ls: List[Dict[str, Any]],
    instruction_ls: List[str]
) -> Dict[str, List[str]]:
    """
    Format final survey output.
    """
    ans_ls = []
    for survey, instruction in zip(survey_ls, instruction_ls):
        if not survey or survey == {}:
            ans_ls.append("No survey generated.")
        else:
            # Format the survey as a readable string
            output = _print_tasknote_hire(survey, last_detail=False)
            ans_ls.append(output)
    
    return {"ans_ls": ans_ls}


if __name__ == "__main__":
    app.run(transport="stdio")

