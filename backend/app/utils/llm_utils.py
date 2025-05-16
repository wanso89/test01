import asyncio
from functools import lru_cache
import torch
import traceback
from typing import Any

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

def generate_text_with_local_llm(
    prompt: str,
    model: Any = None,
    tokenizer: Any = None,
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
    top_p: float = 0.85,
    repetition_penalty: float = 1.1,
) -> str:
    """
    로컬 LLM 모델을 사용하여 텍스트를 생성합니다.
    
    Args:
        prompt: 프롬프트 텍스트
        model: LLM 모델 (None인 경우 전역 모델 사용)
        tokenizer: 토크나이저 (None인 경우 전역 토크나이저 사용)
        temperature: 생성 온도
        max_new_tokens: 최대 새 토큰 수
        top_p: 상위 확률 샘플링 파라미터
        repetition_penalty: 반복 패널티
        
    Returns:
        str: 생성된 텍스트
    """
    if model is None or tokenizer is None:
        print("[LLM_UTILS] 모델 또는 토크나이저가 제공되지 않았습니다. 전역 모델 사용 시도...")
        # 전역 모델과 토크나이저 찾기 시도
        try:
            import sys
            llm_model_module = sys.modules.get("app.main", None)
            if llm_model_module and hasattr(llm_model_module, "llm_model") and hasattr(llm_model_module, "tokenizer"):
                model = llm_model_module.llm_model
                tokenizer = llm_model_module.tokenizer
                print("[LLM_UTILS] 전역 모델 및 토크나이저 로드 성공")
            else:
                print("[LLM_UTILS] 전역 모델 또는 토크나이저를 찾을 수 없습니다.")
                return "텍스트 생성에 필요한 모델을 로드할 수 없습니다."
        except Exception as e:
            print(f"[LLM_UTILS] 전역 모델 로드 오류: {str(e)}")
            return f"텍스트 생성 모델 로드 오류: {str(e)}"
    
    # 모델 및 토크나이저 확인
    if model is None or tokenizer is None:
        return "LLM 모델 또는 토크나이저가 초기화되지 않았습니다."
    
    try:
        print("[LLM_UTILS] 로컬 LLM 모델로 텍스트 생성 시작")
        print(f"[LLM_UTILS] 프롬프트 길이: {len(prompt)} 문자")
        print(f"[LLM_UTILS] 프롬프트 미리보기: {prompt[:120]}...")
        
        # Gemma 모델인지 확인 (모델 이름이나 아키텍처 기반으로 확인)
        model_name = getattr(model.config, "_name_or_path", "").lower()
        is_gemma = "gemma" in model_name
        
        if is_gemma:
            print("[LLM_UTILS] Gemma 모델 감지: 시스템 프롬프트를 직접 프롬프트에 통합")
            # Gemma는 시스템 프롬프트를 "<start_of_turn>system\n시스템 프롬프트<end_of_turn>\n<start_of_turn>user\n사용자 프롬프트<end_of_turn>" 형식으로 통합
            system_prompt = "당신은 정확하고 도움이 되는 인공지능 어시스턴트입니다."
            prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"
        
        print(f"[LLM_UTILS] 최종 프롬프트 길이: {len(prompt)} 문자")
        
        # 비동기 이벤트 루프 충돌 문제 해결 - 동기 방식으로 변경
        # 토큰화 및 모델 입력 형식 변환
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096-max_new_tokens)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 모델 추론 과정을 동기식으로 수행
        with torch.no_grad():
            # 스트리밍이 아닌 일반 생성 방식 사용
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=temperature > 0.0,
            )
            
        # 생성된 전체 시퀀스
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 이후의 생성된 부분만 추출
        generated_text = full_output[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        
        # Gemma 모델인 경우 <end_of_turn> 태그 제거
        if is_gemma and "<end_of_turn>" in generated_text:
            generated_text = generated_text.split("<end_of_turn>")[0].strip()
            
        print(f"[LLM_UTILS] 생성 완료: {len(generated_text)} 문자")
        return generated_text.strip()
        
    except RuntimeError as e:
        error_msg = str(e)
        print(f"[LLM_UTILS] 로컬 LLM 텍스트 생성 중 오류: {error_msg}")
        if "CUDA out of memory" in error_msg:
            return "GPU 메모리 부족으로 텍스트 생성에 실패했습니다. 더 짧은 프롬프트를 사용하거나 관리자에게 문의하세요."
        elif "Cannot run the event loop while another loop is running" in error_msg:
            return "이벤트 루프 충돌 문제가 발생했습니다. 관리자에게 문의하세요."
        else:
            return f"텍스트 생성 중 런타임 오류가 발생했습니다: {error_msg}"
    except Exception as e:
        print(f"[LLM_UTILS] 로컬 LLM 텍스트 생성 중 오류: {str(e)}")
        traceback.print_exc()
        return f"텍스트 생성 중 오류가 발생했습니다: {str(e)}" 