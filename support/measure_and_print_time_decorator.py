from time import perf_counter


def measure_and_print_time_decorator(function):
    def wrapper(*args, **kwargs):
        start_time = perf_counter()

        result = function(*args, **kwargs)  # the function

        end_time = perf_counter()

        elapsed_time = end_time - start_time
        print(f"{function.__name__}: {elapsed_time:.6f} seconds")
        return result
    return wrapper