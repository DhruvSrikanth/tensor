CC = gcc
CFLAGS = -Wall -O3
LDFLAGS = -lm

# turn on all the warnings
# https://github.com/mcinglis/c-style
CFLAGS += -Wall -Wextra -Wpedantic \
          -Wformat=2 -Wno-unused-parameter -Wshadow \
          -Wwrite-strings -Wstrict-prototypes -Wold-style-definition \
          -Wredundant-decls -Wnested-externs -Wmissing-include-dirs

# Main targets
all: tensor libtensor.so

# Compile the main executable
tensor: tensor.c tensor.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Create shared library
libtensor.so: tensor.c tensor.h
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< $(LDFLAGS)

# Clean up build artifacts
clean:
	rm -f tensor libtensor.so

# Test using pytest
test:
	pytest

.PHONY: all clean test tensor
