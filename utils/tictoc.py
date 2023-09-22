import time


def tic():
    # require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()


def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.1f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")
