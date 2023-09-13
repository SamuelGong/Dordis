class PlaintextConst:
    def __init__(self):
        if not hasattr(self, "PREPARE_DATA"):
            self.PREPARE_DATA = 0
            self.ENCODE_DATA = 1
            self.UPLOAD_DATA = 2
            self.GENERATE_OUTPUT = 3
            self.SERVER_USE_OUTPUT = 4
            self.DOWNLOAD_DATA = 5
            self.DECODE_DATA = 6
            self.CLIENT_USE_OUTPUT = 7

            self._no_plot_client_comp = 0
            self._no_plot_comm = 1
            self._no_plot_server_comp = 2
            self._no_plot_phase_stage_mapping = {
                self.PREPARE_DATA: self._no_plot_client_comp,
                self.ENCODE_DATA: self._no_plot_client_comp,
                self.UPLOAD_DATA: self._no_plot_comm,
                self.GENERATE_OUTPUT: self._no_plot_server_comp,
                self.SERVER_USE_OUTPUT: self._no_plot_server_comp,
                self.DOWNLOAD_DATA: self._no_plot_comm,
                self.DECODE_DATA: self._no_plot_client_comp,
                self.CLIENT_USE_OUTPUT: self._no_plot_client_comp
            }

            self._no_plot_combine = 0
            self._no_plot_separate = 1
            self._no_plot_phase_mode_mapping = {
                self.PREPARE_DATA: self._no_plot_combine,
                self.ENCODE_DATA: self._no_plot_separate,
                self.UPLOAD_DATA: self._no_plot_separate,
                self.GENERATE_OUTPUT: self._no_plot_separate,
                self.SERVER_USE_OUTPUT: self._no_plot_separate,
                self.DOWNLOAD_DATA: self._no_plot_separate,
                self.DECODE_DATA: self._no_plot_separate,
                self.CLIENT_USE_OUTPUT: self._no_plot_combine
            }
        # otherwise has already been overridden


class SecAggConst:
    def __init__(self):
        if not hasattr(self, "PREPARE_DATA"):
            self.PREPARE_DATA = 0
            self.ENCODE_DATA = 1
            self.ADVERTISE_KEYS = 2
            self.SHARE_KEYS = 3
            self.MASKING = 4
            self.UPLOAD_DATA = 5
            self.UNMASKING = 6
            self.GENERATE_OUTPUT = 7
            self.SERVER_USE_OUTPUT = 8
            self.DOWNLOAD_DATA = 9
            self.DECODE_DATA = 10
            self.CLIENT_USE_OUTPUT = 11
        # otherwise has already been overridden

            self._no_plot_client_comp = 0
            self._no_plot_comm = 1
            self._no_plot_server_comp = 2
            self._no_plot_phase_stage_mapping = {
                self.PREPARE_DATA: self._no_plot_client_comp,
                self.ENCODE_DATA: self._no_plot_client_comp,
                self.ADVERTISE_KEYS: self._no_plot_client_comp,
                self.SHARE_KEYS: self._no_plot_client_comp,
                self.MASKING: self._no_plot_client_comp,
                self.UPLOAD_DATA: self._no_plot_comm,
                self.UNMASKING: self._no_plot_server_comp,
                self.GENERATE_OUTPUT: self._no_plot_server_comp,
                self.SERVER_USE_OUTPUT: self._no_plot_server_comp,
                self.DOWNLOAD_DATA: self._no_plot_comm,
                self.DECODE_DATA: self._no_plot_client_comp,
                self.CLIENT_USE_OUTPUT: self._no_plot_client_comp
            }

            self._no_plot_combine = 0
            self._no_plot_separate = 1
            self._no_plot_phase_mode_mapping = {
                self.PREPARE_DATA: self._no_plot_combine,
                self.ENCODE_DATA: self._no_plot_separate,
                self.ADVERTISE_KEYS: self._no_plot_separate,
                self.SHARE_KEYS: self._no_plot_separate,
                self.MASKING: self._no_plot_separate,
                self.UPLOAD_DATA: self._no_plot_separate,
                self.UNMASKING: self._no_plot_separate,
                self.GENERATE_OUTPUT: self._no_plot_separate,
                self.SERVER_USE_OUTPUT: self._no_plot_separate,
                self.DOWNLOAD_DATA: self._no_plot_separate,
                self.DECODE_DATA: self._no_plot_separate,
                self.CLIENT_USE_OUTPUT: self._no_plot_combine
            }


class DDGaussConst:
    def __init__(self):
        if not hasattr(self, "PREPARE_DATA"):
            pass
