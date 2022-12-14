import numpy as np
import time
from colored import fg, attr
import random


def print_state(state, prev_state):
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if (state[i, j] == prev_state[i, j]):
                if (state[i, j] == 1):
                    print(fg('sea_green_1b') + "{0: 4d}".format(1) + attr('reset'), end='')

                else:
                    print(fg('deep_pink_2') + "{0: 4d}".format(1) + attr('reset'), end='')

            else:
                print(fg('yellow_1') + "{0: 4d}".format(1) + attr('reset'), end='')

        print()

    print()

def print_matrix(mat):
    print_state(mat, mat)

def activation(x):
    if (x > 0):
        return 1

    return -1

def noise(image):
    positions = [(x, y) for x in range(image.shape[0]) for y in range(image.shape[1])]
    
    noised_count = 20

    for i in range(noised_count):
        pos = random.choice(positions)

        image[pos[0], pos[1]] = - image[pos[0], pos[1]]

    return image

if __name__ == '__main__':
    with open('image.txt') as image:
        content = image.read()

        img_repr = []
        for row in content.split('\n'):
            row_repr = [int(row_item) for row_item in row.split(' ') if row_item != '']
            img_repr.append(row_repr)

        image_height = len(img_repr)
        image_width = len(img_repr[0])

        orig_image = np.expand_dims(np.array(img_repr).flatten(), axis=0)
        noised_image = np.expand_dims(noise(np.array(img_repr)).flatten(), axis=0)

        W_matrix = orig_image.T @ orig_image
        np.fill_diagonal(W_matrix, 0)

        all_cols = set([i for i in range(W_matrix.shape[1])])
        visited = set()

        print("Original image:")
        print_matrix(orig_image.reshape(image_height, image_width))

        print("Noised image:")
        print_matrix(noised_image.reshape(image_height, image_width))

        print("Hopfield relaxation:")

        prev_state = noised_image
        state = np.empty_like(noised_image)

        while True:
            print_state(state.reshape(image_height, image_width), prev_state.reshape(image_height, image_width))
            input()

            col = random.choice(list(all_cols - visited))
            visited.add(col)

            prev_state = state.copy()
            state[0][col] = activation(state @ W_matrix.T[col].T)

            if all_cols == visited and np.sum(state - prev_state, axis=1) == 0:
                break

            elif all_cols == visited:
                visited = {}

        denoised_image = state.reshape(image_height, image_width)
        print(denoised_image)

        time.sleep(2)