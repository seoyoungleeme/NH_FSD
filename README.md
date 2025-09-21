터미널이나 명령 프롬프트를 열고, NH 폴더로 이동한 뒤 다음 명령어를 실행하면 됩니다.


cd path/to/NH_FDS
python main.py

이 구조를 사용하면 main.py가 같은 폴더에 있는 risk_engine.py를 쉽게 import하고, transaction_data.csv 파일을 바로 읽을 수 있어 경로 문제 없이 깔끔하게 실행할 수 있습니다.