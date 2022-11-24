import abc

import click
from nipype.pipeline.engine import Workflow


def postset(attribute, value):
    """Sets the attribute of an object after the execution.
    Args:
        attribute: An object's attribute to be set.
        value: A desired value for the object's attribute.
    Returns:
        A decorator executed after the decorated function is.
    """

    def postset_decorator(func):
        def func_wrapper(self, *args, **kwargs):
            res = func(self, *args, **kwargs)
            setattr(self, attribute, value)
            return res

        return func_wrapper

    return postset_decorator


class Pipeline(Workflow):
    """Clinica Pipeline class.
    This class overwrites the `Workflow` to integrate and encourage the
    use of BIDS and CAPS data structures as inputs and outputs of the pipelines
    developed for the Clinica software.
    The global architecture of a Clinica pipelines is as follow:
        [ Data Input Stream ]
                |
            [ Input ]
                |
            [[ Core ]] <- Could be one or more `npe.Node`s
                |
            [ Output ]
                |
        [ Data Output Stream ]
    Attributes:
        is_built (bool): Informs if the pipelines has been built or not.
        parameters (dict): Parameters of the pipelines.
        info (dict): Information presented in the associated `info.json` file.
        input_node (:obj:`npe.Node`): Identity interface connecting inputs.
        output_node (:obj:`npe.Node`): Identity interface connecting outputs.
        bids_directory (str): Directory used to read the data from, in BIDS.
        caps_directory (str): Directory used to read/write the data from/to,
            in CAPS.
        subjects (list): List of subjects defined in the `subjects.tsv` file.
            # TODO(@jguillon): Check the subjects-sessions file name.
        sessions (list): List of sessions defined in the `subjects.tsv` file.
        tsv_file (str): Path to the subjects-sessions `.tsv` file.
        info_file (str): Path to the associated `info.json` file.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
            self,
            bids_directory=None,
            caps_directory=None,
            tsv_file=None,
            overwrite_caps=False,
            base_dir=None,
            parameters={},
            name=None,
    ):
        """Init a Pipeline object.
        Args:
            bids_directory (str, optional): Path to a BIDS directory. Defaults to None.
            caps_directory (str, optional): Path to a CAPS directory. Defaults to None.
            tsv_file (str, optional): Path to a subjects-sessions `.tsv` file. Defaults to None.
            overwrite_caps (bool, optional): Overwrite or not output directory.. Defaults to False.
            base_dir (str, optional): Working directory (attribute of Nipype::Workflow class). Defaults to None.
            parameters (dict, optional): Pipeline parameters. Defaults to {}.
            name (str, optional): Pipeline name. Defaults to None.
        Raises:
            RuntimeError: [description]
        """
        import inspect
        import os
        from tempfile import mkdtemp

        from clinica.utils.inputs import check_bids_folder, check_caps_folder
        from clinica.utils.participant import get_subject_session_list

        self._is_built = False
        self._overwrite_caps = overwrite_caps
        self._bids_directory = bids_directory
        self._caps_directory = caps_directory
        self._verbosity = "debug"
        self._tsv_file = tsv_file
        self._info_file = os.path.join(
            os.path.dirname(os.path.abspath(inspect.getfile(self.__class__))),
            "info.json",
        )
        self._info = {}

        if base_dir:
            self.base_dir = base_dir
            self._base_dir_was_specified = True
        else:
            self.base_dir = mkdtemp()
            self._base_dir_was_specified = False

        self._name = name or self.__class__.__name__
        self._parameters = parameters

        if not self._bids_directory:
            if not self._caps_directory:
                raise RuntimeError(
                    f"The {self._name} pipeline does not contain "
                    "BIDS nor CAPS directory at the initialization."
                )

            check_caps_folder(self._caps_directory)
            input_dir = self._caps_directory
            is_bids_dir = False
        else:
            check_bids_folder(self._bids_directory)
            input_dir = self._bids_directory
            is_bids_dir = True
        self._sessions, self._subjects = get_subject_session_list(
            input_dir, tsv_file, is_bids_dir, False, base_dir
        )

        self.init_nodes()

    @staticmethod
    def get_processed_images(caps_directory, subjects, sessions):
        """Extract processed image IDs in `caps_directory` based on `subjects`_`sessions`.
        Todo:
            [ ] Implement this static method in all pipelines
            [ ] Make it abstract to force overload in future pipelines
        """
        from clinica.utils.exceptions import ClinicaException
        from clinica.utils.stream import cprint

        cprint(msg="Pipeline finished with errors.", lvl="error")
        cprint(msg="CAPS outputs were not found for some image(s):", lvl="error")
        raise ClinicaException(
            "Implementation on which image(s) failed will appear soon."
        )

    def init_nodes(self):
        """Init the basic workflow and I/O nodes necessary before build."""
        import nipype.interfaces.utility as nutil
        import nipype.pipeline.engine as npe

        if self.get_input_fields():
            self._input_node = npe.Node(
                name="Input",
                interface=nutil.IdentityInterface(
                    fields=self.get_input_fields(), mandatory_inputs=False
                ),
            )
        else:
            self._input_node = None

        if self.get_output_fields():
            self._output_node = npe.Node(
                name="Output",
                interface=nutil.IdentityInterface(
                    fields=self.get_output_fields(), mandatory_inputs=False
                ),
            )
        else:
            self._output_node = None

        Workflow.__init__(self, self._name, self.base_dir)
        if self.input_node:
            self.add_nodes([self.input_node])
        if self.output_node:
            self.add_nodes([self.output_node])

    def has_input_connections(self):
        """Checks if the Pipeline's input node has been connected.
        Returns:
            True if the input node is connected, False otherwise.
        """
        if self.input_node:
            return self._graph.in_degree(self.input_node) > 0
        else:
            return False

    def has_output_connections(self):
        """Checks if the Pipeline's output node has been connected.
        Returns:
            True if the output node is connected, False otherwise.
        """
        if self.output_node:
            return self._graph.out_degree(self.output_node) > 0
        else:
            return False

    @postsetp("is_built", True)
    def build(self):
        """Builds the core, input and output nodes of the Pipeline.
        This method first checks it has already been run. It then checks
        the pipelines dependencies and, in this order, builds the core nodes,
        the input node and, finally, the output node of the Pipeline.
        Since this method returns the concerned object, it can be chained to
        any other method of the Pipeline class.
        Returns:
            self: A Pipeline object.
        """
        if not self.is_built:
            self.check_dependencies()
            self.check_pipeline_parameters()
            if not self.has_input_connections():
                self.build_input_node()
            self.build_core_nodes()
            if not self.has_output_connections():
                self.build_output_node()
        return self

    def run(self, plugin=None, plugin_args=None, update_hash=False, bypass_check=False):
        """Executes the Pipeline.
        It overwrites the default Workflow method to check if the
        Pipeline is built before running it. If not, it builds it and then
        run it.
        It also checks whether there is enough space left on the disks, and if
        the number of threads to run in parallel is consistent with what is
        possible on the CPU.
        Args:
            Similar to those of Workflow.run.
        Returns:
            An execution graph (see Workflow.run).
        """

        plugin_args = self.update_parallelize_info(plugin_args)
        plugin = "MultiProc"
        exec_graph = Workflow.run(self, plugin, plugin_args, update_hash)

        return exec_graph

    def update_parallelize_info(self, plugin_args):
        """Performs some checks of the number of threads given in parameters,
        given the number of CPUs of the machine in which clinica is running.
        We force the use of plugin MultiProc
        Author: Arnaud Marcoux"""
        import select
        import sys
        from multiprocessing import cpu_count

        from clinica.utils.stream import cprint

        # count number of CPUs
        n_cpu = cpu_count()
        # timeout value: max time allowed to decide how many thread
        # to run in parallel (sec)

        # Use this var to know in the end if we need to ask the user
        # an other number

        try:
            # if no --n_procs arg is used, plugin_arg is None
            # so we need a try / except block
            n_thread_cmdline = plugin_args["n_procs"]
            if n_thread_cmdline > n_cpu:
                raise Exception(
                    f"You are trying to run clinica with a number of threads ({n_thread_cmdline}) superior to your "
                    f"number of CPUs ({n_cpu}).")

        except TypeError:
            raise Exception(f"You did not specify the number of threads to run in parallel (--n_procs argument).")

        if plugin_args:
            plugin_args["n_procs"] = n_procs
        else:
            plugin_args = {"n_procs": n_procs}

        return plugin_args
