"""
Qwen3 모델용 최적화된 프롬프트 템플릿 모듈
"""

from typing import List, Dict, Any, Optional

def create_chat_prompt(
    question: str, 
    context: str, 
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    use_html: bool = True,
    language: str = "ko"
) -> str:
    """
    Qwen3 모델에 최적화된 RAG 챗봇 프롬프트 템플릿을 생성합니다.
    
    Args:
        question: 사용자 질문
        context: 검색된 문서 컨텍스트
        conversation_history: 대화 기록 (선택 사항)
        use_html: HTML 태그 허용 여부
        language: 응답 언어 (기본값: 한국어)
        
    Returns:
        str: 포맷팅된 프롬프트
    """
    # 대화 기록 처리
    conversation_context = ""
    if conversation_history:
        conversation_parts = []
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        for msg in recent_history:
            role = "사용자" if msg["role"] == "user" else "시스템"
            conversation_parts.append(f"{role}: {msg['content']}")
        conversation_context = "대화 기록:\n" + "\n".join(conversation_parts) + "\n\n"
    
    # HTML 태그 관련 지시사항
    html_instruction = "<b>, <ul>, <li>" if use_html else "사용하지 마세요"
    
    # 언어별 시스템 프롬프트
    if language == "ko":
        system_content = f"""당신은 사용자 질문에 대해 주어진 참고 문서를 기반으로 답변하는 한국어 AI 어시스턴트입니다.
다음 지침을 매우 엄격히 따라주세요:

1. 반드시 한국어로만 답변하세요.
2. 답변은 반드시 제공된 '참고 문서' 섹션의 내용에 근거해야 합니다.
3. 문서에 질문과 관련된 정보가 없다면, "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 명확히 답변하세요.
4. 답변은 명확하고 간결하게 작성해주세요.
5. HTML 태그는 {html_instruction}만 사용할 수 있습니다."""
    else:
        # 다국어 지원을 위한 기본 영어 템플릿
        system_content = f"""You are an AI assistant that answers user questions based on the provided reference documents.
Please strictly follow these guidelines:

1. Answer in the specified language only.
2. Base your answers solely on the information in the 'Reference Documents' section.
3. If the documents don't contain relevant information, clearly state "I cannot find relevant information in the provided documents."
4. Provide clear and concise answers.
5. HTML tags: You may only use {html_instruction}."""

    # 최종 프롬프트 조합
    prompt = f"""<|im_start|>system
{system_content}
<|im_end|>

<|im_start|>user
{conversation_context}현재 질문: {question}

참고 문서:
{context}
<|im_end|>

<|im_start|>assistant
"""
    
    return prompt

def create_retrieval_improvement_prompt(query: str, relevant_chunks: List[str], irrelevant_chunks: List[str]) -> str:
    """
    검색 개선을 위한 프롬프트 생성
    
    Args:
        query: 원본 쿼리
        relevant_chunks: 관련성 높은 청크 목록
        irrelevant_chunks: 관련성 낮은 청크 목록
        
    Returns:
        str: 검색 개선 프롬프트
    """
    # 문서 조각들을 미리 처리하여 백슬래시 문제 해결
    relevant_docs = '\n---\n'.join(relevant_chunks[:3])
    irrelevant_docs = '\n---\n'.join(irrelevant_chunks[:3])
    
    return f"""<|im_start|>system
당신은 정보 검색 시스템의 검색 품질을 개선하는 AI 전문가입니다. 사용자 질문에 대해 더 관련성 높은 검색 결과를 반환하도록 쿼리를 개선해주세요.
<|im_end|>

<|im_start|>user
원본 사용자 질문: {query}

관련성 높다고 판단된 문서:
{relevant_docs}

관련성 낮다고 판단된 문서:
{irrelevant_docs}

다음을 수행해주세요:
1. 원본 질문을 분석하여 핵심 키워드와 의도를 파악하세요.
2. 관련성 높은 문서에서 중요 용어와 개념을 추출하세요.
3. 관련성 낮은 문서를 분석하여 잘못된 방향으로 검색된 이유를 파악하세요.
4. 원본 질문을 수정하여 더 정확한 검색 결과를 얻을 수 있는 개선된 쿼리를 제안하세요.

개선된 쿼리만 JSON 형식으로 제공하세요: {{"improved_query": "개선된 쿼리 내용"}}
<|im_end|>

<|im_start|>assistant
"""

def create_document_summarization_prompt(document_text: str, max_length: int = 300) -> str:
    """
    문서 요약 프롬프트 생성
    
    Args:
        document_text: 요약할 문서 텍스트
        max_length: 최대 요약 길이 (기본값: 300자)
        
    Returns:
        str: 문서 요약 프롬프트
    """
    return f"""<|im_start|>system
당신은 문서 내용을 명확하고 간결하게 요약하는 AI 전문가입니다. 주어진 문서의 핵심 내용을 유지하면서 간결한 요약을 제공해주세요.
<|im_end|>

<|im_start|>user
다음 문서 내용을 {max_length}자 이내로 요약해주세요. 핵심 정보만 포함하고, 중요하지 않은 세부 사항은 생략하세요.

문서 내용:
{document_text}
<|im_end|>

<|im_start|>assistant
"""

def create_title_generation_prompt(conversation_history: List[Dict[str, Any]]) -> str:
    """
    대화 제목 생성 프롬프트
    
    Args:
        conversation_history: 대화 기록
        
    Returns:
        str: 대화 제목 생성 프롬프트
    """
    # 대화 기록에서 최대 3개 대화만 포함
    conversation_text = ""
    max_turns = min(len(conversation_history), 3)
    
    for i in range(max_turns):
        msg = conversation_history[i]
        role = "사용자" if msg["role"] == "user" else "AI"
        content = msg["content"]
        # 긴 메시지는 100자로 제한
        if len(content) > 100:
            content = content[:97] + "..."
        conversation_text += f"{role}: {content}\n\n"
    
    return f"""<|im_start|>system
당신은 대화 내용을 분석하여 간결하고 명확한 제목을 생성하는 AI 전문가입니다.
<|im_end|>

<|im_start|>user
다음 대화 내용을 분석하여 15자 이내의 간결한 제목을 생성해주세요:

{conversation_text}

제목은 대화의 핵심 주제나 질문을 반영해야 합니다. 불필요한 단어는 생략하고 핵심만 포함하세요.
<|im_end|>

<|im_start|>assistant
""" 