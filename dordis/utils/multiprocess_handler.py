import logging
import os
import time
from abc import abstractmethod, ABC
from multiprocessing import Process, \
    get_start_method, set_start_method
from dordis.utils.share_memory_handler import ShareBase
# from dordis.utils.cpu_affinity import CPUAffinity


# class HandlerProcess(Process, CPUAffinity):
class HandlerProcess(Process):
    def __init__(self, obj, routine, aux,
                 args, delay_mocking_factor=None):
        Process.__init__(self)
        # CPUAffinity.__init__(self)

        self.obj = obj
        self.routine = routine
        self.args = args
        self.aux = aux
        self.delay_mocking_factor = delay_mocking_factor

    def run(self):
        # if self.cpu_affinity_dict is not None:
        #     self.set_cpu_affinity(self.cpu_affinity_dict["comp"])

        self.obj.handler_head(self.aux)
        method_to_call = getattr(self.obj, self.routine)

        start_time = time.perf_counter()
        response = method_to_call(self.args)
        duration = time.perf_counter() - start_time
        if self.delay_mocking_factor is not None:
            delay = self.delay_mocking_factor * duration
            delay = round(delay, 2)
            if delay > 0:
                logging.info(f"[#{os.getpid()}] Waiting another {delay} seconds "
                             f"for mocking computing delay.")
                time.sleep(delay)
                logging.info(f"[#{os.getpid()}] Computing delay mocked.")

        self.obj.handler_tail(self.aux, response)


class MPBase(ShareBase, ABC):
    def __init__(self, client_id):
        super(MPBase, self).__init__(client_id=client_id)

    def get_process_key(self, aux):
        if isinstance(aux, tuple):
            return "_".join([str(i) for i in aux])
        else:
            return None

    def set_log(self):
        logging.basicConfig(
            format=
            '[%(levelname)s][%(asctime)s.%(msecs)03d] '
            '[%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO,
            datefmt='(%Y-%m-%d) %H:%M:%S')

    def spawn_to_handle(self, aux, routine, args,
                        delay_mocking_factor=None):
        if get_start_method(allow_none=True) != 'spawn':
            set_start_method('spawn', force=True)
        p = HandlerProcess(obj=self, routine=routine,
                           aux=aux, args=args,
                           delay_mocking_factor=delay_mocking_factor)
        p.start()

    @abstractmethod
    def handler_head(self, aux):
        """ """

    @abstractmethod
    def handler_tail(self, aux, response):
        """ """


class CustomizedMPBase(MPBase, ABC):
    """ Tailored for this project. """
    def __init__(self, client_id):
        super(CustomizedMPBase, self).__init__(client_id=client_id)
        self.client_id = client_id
        self.status_prefix = "status"

    def get_log_prefix_str(self, round_idx=None, chunk_idx=None,
                           phase_idx=None, pid=None,
                           logical_client_id=None):
        if pid is None:
            pid = os.getpid()

        if logical_client_id is None:
            entity = f"[Server #{pid}]"
        else:
            entity = f"[Client #{logical_client_id} " \
                     f"@ #{self.client_id} #{pid}]"

        if round_idx is not None \
                and chunk_idx is not None \
                and phase_idx is not None:
            return f"{entity} " \
                   f"[Round {round_idx}] " \
                   f"[Chunk {chunk_idx}] " \
                   f"[Phase {phase_idx}]"
        else:
            return f"{entity}"

    def set_status_prefix(self, prefix):
        self.status_prefix = prefix

    def get_a_status(self, key):
        return self.get_a_shared_value(
            key=[self.status_prefix, key])

    def set_a_status(self, key, value):
        self.set_a_shared_value(
            key=[self.status_prefix, key],
            value=value
        )

    def delete_a_status(self, key):
        self.delete_a_shared_value(key=[self.status_prefix, key])

    def get_status_dict(self):
        return self.prefix_to_dict(
            prefix=[self.status_prefix],
            key_type='str'
        )
