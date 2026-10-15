##
## EPITECH PROJECT, 2022
## Makefile
## File description:
## task01
##
SRC =

OBJ = $(SRC.c=.o)

NAME = name

all : $(NAME)

$(NAME) :	$(OBJ)
	make -C generator
	make -C analyzer

clean :
	make clean -C generator
	make clean -C analyzer

fclean : clean
	make fclean -C generator
	make fclean -C analyzer

re : fclean all

.PHONY: re clean fclean all
