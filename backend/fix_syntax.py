#!/usr/bin/env python3
import re

# 파일 경로
file_path = "app/main.py"

# 파일 내용 읽기
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 문제 1: LangchainEmbeddingFunction 클래스 들여쓰기 수정
# 클래스 정의를 올바르게 들여쓰기 수정
pattern1 = r'(\s+)class LangchainEmbeddingFunction:'
replacement1 = r'        class LangchainEmbeddingFunction:'
content = re.sub(pattern1, replacement1, content)

# 문제 2: try-except 문법 오류 수정
pattern2 = r'except Exception as e:(\s+)print\(f"임베딩 생성 오류: \{e\}"\)(\s+)traceback\.print_exc\(\)'
replacement2 = r'except Exception as e:\n                    print(f"임베딩 생성 오류: {e}")\n                    traceback.print_exc()'
content = re.sub(pattern2, replacement2, content)

# 문제 3: query_embedding 들여쓰기 수정
pattern3 = r'if callable\(self\.embedding_function\):(\s+)query_embedding'
replacement3 = r'if callable(self.embedding_function):\n                    query_embedding'
content = re.sub(pattern3, replacement3, content)

# 수정된 내용을 파일에 쓰기
with open(file_path, 'w', encoding='utf-8') as file:
    file.write(content)

print("문법 오류가 수정되었습니다. app/main.py 파일을 확인하세요.") 