CODE_DIR = src

all: $(CODE_DIR)/main.py
	chmod +x downloadGithub.sh
	./downloadGithub.sh
	python3.6 $(CODE_DIR)/main.py
