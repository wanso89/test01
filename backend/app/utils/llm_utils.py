import asyncio
from functools import lru_cache
import torch
import traceback

# 기존 함수들 유지하면서 아래 함수 개선

async def generate_text_with_llm(model, tokenizer, prompt, temperature=0.1, max_tokens=1000):
    """
    LLM 모델을 사용하여 직접 텍스트를 생성합니다.
    
    Args:
        model: LLM 모델
        tokenizer: 토크나이저
        prompt: 프롬프트 텍스트
        temperature: 생성 다양성 조절 (0에 가까울수록 결정적)
        max_tokens: 생성할 최대 토큰 수
    
    Returns:
        생성된 텍스트
    """
    try:
        # 모델과 토크나이저 확인
        if model is None or tokenizer is None:
            return "모델 또는 토크나이저가 초기화되지 않았습니다."
        
        print(f"LLM 텍스트 생성 시작 (프롬프트 길이: {len(prompt)} 문자)")
        
        # 메모리 최적화
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 모델 디바이스 확인 및 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        print(f"모델 디바이스: {device}, 토크나이저 모델 이름: {tokenizer.name_or_path if hasattr(tokenizer, 'name_or_path') else '알 수 없음'}")
            
        with torch.no_grad():
            # 입력 인코딩 및 장치로 이동
            try:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000)
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                print(f"입력 인코딩 중 오류: {e}")
                traceback.print_exc()
                return f"입력 처리 중 오류가 발생했습니다: {str(e)}"
            
            try:
                # 생성 파라미터 설정 및 타임아웃 적용
                outputs = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.generate,
                        **inputs,
                        max_new_tokens=max_tokens,
                        repetition_penalty=1.2,
                        temperature=temperature,
                        do_sample=(temperature > 0.01),  # 온도가 매우 낮을 때는 샘플링 비활성화
                        top_p=0.95,
                        top_k=50,
                        pad_token_id=tokenizer.eos_token_id,
                        num_beams=1,  # 빔 서치 없이 빠른 생성
                    ),
                    timeout=60.0  # 60초 타임아웃 설정
                )
                
                print(f"LLM 응답 생성 완료 (출력 길이: {len(outputs[0])} 토큰)")
            except asyncio.TimeoutError:
                print("LLM 응답 생성 시간 초과")
                return "응답 생성 시간이 초과되었습니다. 질문을 더 간단하게 해주세요."
            except Exception as e:
                print(f"모델 생성 중 오류: {e}")
                traceback.print_exc()
                return f"모델 실행 중 오류가 발생했습니다: {str(e)}"
            
            try:
                # 생성된 텍스트 디코딩
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"디코딩된 텍스트 길이: {len(generated_text)} 문자")
            except Exception as e:
                print(f"텍스트 디코딩 중 오류: {e}")
                traceback.print_exc()
                return f"텍스트 디코딩 중 오류가 발생했습니다: {str(e)}"
            
            # 프롬프트 제거 시도 (응답만 추출)
            if prompt in generated_text:
                response = generated_text[len(prompt):].strip()
                print("프롬프트 정확히 일치. 제거 성공")
            else:
                # 부분적 일치 시도
                prompt_parts = prompt.split('\n\n')
                for part in prompt_parts:
                    if part and part in generated_text and len(part) > 50:
                        idx = generated_text.find(part) + len(part)
                        response = generated_text[idx:].strip()
                        print(f"프롬프트 부분 일치({len(part)} 문자). 해당 부분 이후 추출")
                        break
                else:
                    # 일반적인 마커 탐색
                    for marker in ["답변:", "답변은 다음과 같습니다:", "응답:", "결과:", "위 정보를 바탕으로"]:
                        if marker in generated_text:
                            response = generated_text.split(marker, 1)[1].strip()
                            print(f"마커('{marker}') 발견. 마커 이후 추출")
                            break
                    else:
                        # 모든 방법 실패 시 전체 텍스트 사용
                        response = generated_text
                        print("프롬프트 제거 실패. 전체 텍스트 사용")
            
            # 'model' 접두어 제거 (LLM 응답에서 종종 발생)
            if response.startswith("model") and len(response) > 5:
                response = response[5:].strip()
                
            # 응답 길이 확인 및 출력
            print(f"최종 응답 길이: {len(response)} 문자")
            print(f"응답 미리보기: {response[:100]}..." if len(response) > 100 else f"응답: {response}")
                
            return response
    except Exception as e:
        print(f"텍스트 생성 중 예상치 못한 오류 발생: {e}")
        traceback.print_exc()
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"

async def generate_response_async(model, tokenizer, prompt, max_tokens=1000, temperature=0.1):
    """
    LLM 모델을 사용하여 프롬프트에 대한 응답을 생성합니다.
    """
    from app.utils.generator import generate_llm_response
    
    # 기존 LLM 응답 생성 함수 활용
    return await generate_llm_response(
        llm_model=model,
        tokenizer=tokenizer,
        question=prompt,
        top_docs=[],  # 검색 결과 없이 직접 질문에 답변
        temperature=temperature,
        max_tokens=max_tokens
    )

def generate_response(prompt, max_tokens=1000, temperature=0.1):
    """
    비동기 함수를 동기적으로 호출하기 위한 래퍼 함수입니다.
    """
    import fastapi
    from app.main import app
    
    # FastAPI 앱 객체에서 모델과 토크나이저 가져오기
    model = app.state.llm_model
    tokenizer = app.state.tokenizer
    
    # 비동기 루프 생성 및 함수 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    response = loop.run_until_complete(
        generate_response_async(model, tokenizer, prompt, max_tokens, temperature)
    )
    loop.close()
    
    return response 

def generate_text_with_local_llm(prompt, max_tokens=1024, temperature=0.5, top_p=0.95, top_k=50, system_prompt=None):
    """
    로컬 LLM 모델을 사용하여 텍스트를 생성하는 함수입니다.
    
    Args:
        prompt (str): 프롬프트 텍스트
        max_tokens (int): 생성할 최대 토큰 수 (기본값 1024로 증가)
        temperature (float): 생성 다양성 조절 (0.5로 증가하여 더 창의적인 응답)
        top_p (float): 누적 확률 임계값 (nucleus sampling)
        top_k (int): 상위 k개 토큰만 고려
        system_prompt (str): 시스템 프롬프트 (지정하지 않으면 기본값 사용)
        
    Returns:
        str: 생성된 텍스트
    """
    import fastapi
    from app.main import app
    
    print(f"[LLM_UTILS] 로컬 LLM 모델로 텍스트 생성 시작")
    print(f"[LLM_UTILS] 프롬프트 길이: {len(prompt)} 문자")
    print(f"[LLM_UTILS] 프롬프트 미리보기: {prompt[:100]}...")
    
    # 기본 시스템 프롬프트 설정
    if system_prompt is None:
        system_prompt = """당신은 전문적인 AI 어시스턴트입니다. 다음 지침을 항상 따르세요:
1. 질문에 대해 정확하고 상세한 답변을 제공하세요.
2. 핵심 정보를 단순하게 요약하는 대신, 풍부한 맥락과 설명을 제공하세요.
3. 사용자가 요청한 내용을 깊이 있게 분석하고, 관련 정보를 포괄적으로 다루세요.
4. 답변은 명확한 구조로 논리적으로 구성하고, 필요한 세부 사항과 예시를 포함하세요.
5. 단순한 일반화나 피상적인 응답은 피하고, 구체적이고 실용적인 통찰을 제공하세요.
6. 항상 사용자의 질문 의도를 정확히 파악하여 해당 주제에 대한 전문적인 지식을 보여주세요."""
    
    # 생성 지시사항 추가하여 프롬프트 강화
    enhanced_prompt = f"""{prompt}

답변 지침:
- 응답은 반드시 충분히 길고 상세하게 작성해주세요.
- 단답형으로 응답하지 말고 충분한 설명과 분석을 포함해주세요.
- 필요한 경우 예시를 들거나 단계별로 설명해주세요.
- 주제에 관한 중요한 세부 정보를 빠짐없이 포함시켜주세요."""
    
    # FastAPI 앱 객체에서 모델과 토크나이저 가져오기
    model = getattr(app.state, "llm_model", None)
    tokenizer = getattr(app.state, "tokenizer", None)
    
    if model is None or tokenizer is None:
        print("[LLM_UTILS] 모델 또는 토크나이저가 초기화되지 않았습니다.")
        return "모델 준비 중입니다. 잠시 후 다시 시도해주세요."
    
    try:
        # 비동기 루프 생성 및 함수 실행
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # 모델마다 시스템 프롬프트 적용 방식 다를 수 있음 (Gemma는 시스템 프롬프트를 따로 지원하지 않음)
        # 실제 모델이 Gemma인 경우 시스템 프롬프트를 프롬프트 앞에 추가
        if "gemma" in str(model.__class__).lower() or "gemma" in (getattr(tokenizer, "name_or_path", "").lower()):
            print("[LLM_UTILS] Gemma 모델 감지: 시스템 프롬프트를 직접 프롬프트에 통합")
            full_prompt = f"{system_prompt}\n\n사용자: {enhanced_prompt}\n\n응답:"
        else:
            # 다른 모델들은 generate_text_with_llm 내부에서 시스템 프롬프트를 적용할 것으로 가정
            full_prompt = enhanced_prompt
            
        print(f"[LLM_UTILS] 최종 프롬프트 길이: {len(full_prompt)} 문자")
        
        response = loop.run_until_complete(
            generate_text_with_llm(
                model=model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )
        loop.close()
        
        print(f"[LLM_UTILS] 응답 길이: {len(response)} 문자")
        print(f"[LLM_UTILS] 응답 미리보기: {response[:100]}...")
        
        # 응답이 너무 짧은 경우 오류 피드백
        if len(response) < 20:
            print(f"[LLM_UTILS] 경고: 응답이 너무 짧습니다 - '{response}'")
            if "error" in response.lower() or "오류" in response:
                return response
            return f"답변을 생성하는 데 문제가 발생했습니다. 응답이 너무 짧습니다: {response}"
        
        return response
    except Exception as e:
        print(f"[LLM_UTILS] 로컬 LLM 텍스트 생성 중 오류: {e}")
        traceback.print_exc()
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}" 