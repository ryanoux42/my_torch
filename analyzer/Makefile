##
## EPITECH PROJECT, 2024
## B-CNA-410-PAR-4-1-my_pgp-ryan.hercule
## File description:
## Makefile
##

SRC	        =	my_torch_analyzer.py

CC 		    =	cp

NAME		=	../my_torch_analyzer 

all : $(NAME)

$(NAME) :
	$(CC) $(SRC) $(NAME)
	chmod +x $(NAME)

clean:
	rm -rf __pycache__

fclean: clean
	rm -f $(NAME)

re: fclean all

.PHONY: all clean fclean
