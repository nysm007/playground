# -*-coding:utf-8 -*-

import sys
import os
import time
import datetime
import traceback
import gzip
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait
from multiprocessing import Process, Manager, Pool, cpu_count


def get_files(path):
    output = list()
    for root, sub_dir, files in os.walk(path):
        for filename in files:
            file_abs_path = os.path.join(root, filename)
            output.append(file_abs_path)
    return output


def surfix_filter(files, surfix):
    output = list()
    for filename in files:
        if filename.endswith(surfix):
            output.append(filename)
    return output


def dummy_function(some_content):
    print("Current content [{}] time: {}".format(some_content, time.time()))


def one_process_n_thread(args, thread_num):
    thread_pool = ThreadPoolExecutor(thread_num)
    futures = list()
    pid = os.getpid()
    print("Current pid: {}".format(pid))

    for something in args:
        futures.append(thread_pool.submit(dummy_function, something))


def main_process(filename, global_pid_set):
    global_pid_set[os.getpid()] = time.time()
    date_dict = dict()
    exception_lines = list()
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            try:
                splitted = line.split(",")
                date = splitted[19].strip()  # should be configurable
                if date not in date_dict:
                    date_dict[date] = list()
                new_line = line + "hahhaha" + "\n"  # TODO
                date_dict[date].append(new_line)
            except Exception:
                exception_lines.append(traceback.print_exc() + "\n")
                exception_lines.append(line + "\n")

    print("TEST")
    exception_file = "exception.{}".format(filename)
    with open("exception.{}".format(exception_file), "w") as f:
        f.writelines(exception_lines)

    print("TEST2")

    for k, v in date_dict.items():
        new_filename = "rawlog.{}_0000.{}.gz".format(k, filename)
        with gzip.open(new_filename, "w") as f:

            for line in v:
                f.write(line)
        print("Dump finished for {} at {}.".format(new_filename, str(datetime.datetime.now())))
    del global_pid_set[os.getpid()]


if __name__ == "__main__":
    start_time = time.time()
    print("Parent process id: {}".format(os.getpid()))
    path = sys.argv[1]
    csv_files = surfix_filter(get_files(path), ".csv")

    print("Processing files in total: {}".format(len(csv_files)))
    num_cores = cpu_count() * 2
    print("Using {} cores/sub-processes to run the job.".format(num_cores))

    pool = Pool(num_cores)
    manager = Manager()
    global_pid_set = manager.dict()
    for filename in csv_files:
        pool.apply_async(main_process, args=(filename, global_pid_set, ))
    print("Wait for all sub-process done...")

    pool.close()
    pool.join()
    print("All sub-process done, time consumed: {}".format(time.time() - start_time))
