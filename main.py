import argparse

import experiments
import stag.graph


class PerfData(object):
    """
    An object storing performance data for a single run of a
    clustering algorithm.
    """

    def __init__(self, g: stag.graph.Graph,
                 ari: float, nmi: float, f1: float, t: float,
                 ari_std=None, nmi_std=None, f1_std=None,
                 t_std=None):
        """
        :param ari: The ARI of the returned clustering.
        :param t: The time taken by the clustering algorithm.
        """
        self.n = g.number_of_vertices()
        self.ari = ari
        self.nmi = nmi
        self.f1 = f1
        self.time = t
        self.ari_std = ari_std
        self.nmi_std = nmi_std
        self.f1_std = f1_std
        self.t_std = t_std


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument('command', type=str, choices=['plot', 'run'])
    parser.add_argument('experiment', type=str)
                        # choices=['fig2a', 'fig2b', 'mnist', 'pen', 'fashion',
                        #          'har', 'letter'])
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'plot':
        if args.experiment not in ['fig2a', 'fig2b']:
            print("Can only plot SBM experiments. Specify 'fig2a' or 'fig2b'.")
        elif args.experiment == 'fig2a':
            experiments.sbm_plot("grow_k")
        else:
            experiments.sbm_plot("grow_n")
    else:
        # if args.experiment == "fig2a":
        #     experiments.run_sbm_experiment_growing_k()
        # elif args.experiment == "fig2b":
        #     experiments.run_sbm_experiment_growing_n()
        # elif args.experiment == "mnist":
        #     experiments.openml_experiment("mnist_784", t_const=15)
        # elif args.experiment == "fashion":
        #     experiments.openml_experiment("Fashion-MNIST", t_const=15)
        # elif args.experiment == "har":
        #     experiments.openml_experiment("har", t_const=30)
        # elif args.experiment == "letter":
        #     experiments.openml_experiment("letter", t_const=15)
        # elif args.experiment == "pen":
        #     experiments.openml_experiment("pendigits", t_const=30)
        if args.experiment == "spiral":
            experiments.openml_experiment("spiral", t_const=15)
        elif args.experiment == "4C":
            experiments.openml_experiment("4C", t_const=15)
        elif args.experiment == "AC":
            experiments.openml_experiment("AC", t_const=15)
        elif args.experiment == "RingG":
            experiments.openml_experiment("RingG", t_const=15)
        elif args.experiment == "mnist100000":
            experiments.openml_experiment("mnist100000", t_const=15)
        elif args.experiment == "landsat":
            experiments.openml_experiment("landsat", t_const=15)
        elif args.experiment == "complex9":
            experiments.openml_experiment("complex9", t_const=15)
        elif args.experiment == "cure-t2-4k":
            experiments.openml_experiment("cure-t2-4k", t_const=15)
        elif args.experiment == "spam":
            experiments.openml_experiment("spam", t_const=30)
        elif args.experiment == "waveform3":
            experiments.openml_experiment("waveform3", t_const=30)
        elif args.experiment == "pendigits":
            experiments.openml_experiment("pendigits", t_const=30)
        elif args.experiment == "USPS":
            experiments.openml_experiment("USPS", t_const=30)
        elif args.experiment == "letters":
            experiments.openml_experiment("letters", t_const=30)
        elif args.experiment == "mnist":
            experiments.openml_experiment("mnist", t_const=30)
        elif args.experiment == "skin":
            experiments.openml_experiment("skin", t_const=30)
        elif args.experiment == "covertype":
            experiments.openml_experiment("covertype", t_const=30)
        elif args.experiment == "dense_3_sparse_3_sparse_3":
            experiments.openml_experiment("dense_3_sparse_3_sparse_3", t_const=15)
        elif args.experiment == "dense_8_sparse_1_sparse_1":
            experiments.openml_experiment("dense_8_sparse_1_sparse_1", t_const=15)
        elif args.experiment == "one_gaussian_10_one_line_5_2":
            experiments.openml_experiment("one_gaussian_10_one_line_5_2", t_const=15)

        elif args.experiment == "usps_with_0.1_noise":
            experiments.openml_experiment("usps_with_0.1_noise", t_const=30)
        elif args.experiment == "usps_with_0.02_noise":
            experiments.openml_experiment("usps_with_0.02_noise", t_const=30)
        elif args.experiment == "usps_with_0.2_noise":
            experiments.openml_experiment("usps_with_0.2_noise", t_const=30)
        elif args.experiment == "usps_with_0.04_noise":
            experiments.openml_experiment("usps_with_0.04_noise", t_const=30)
        elif args.experiment == "usps_with_0.06_noise":
            experiments.openml_experiment("usps_with_0.06_noise", t_const=30)
        elif args.experiment == "usps_with_0.08_noise":
            experiments.openml_experiment("usps_with_0.08_noise", t_const=30)
        elif args.experiment == "usps_with_0.12_noise":
            experiments.openml_experiment("usps_with_0.12_noise", t_const=30)
        elif args.experiment == "usps_with_0.14_noise":
            experiments.openml_experiment("usps_with_0.14_noise", t_const=30)
        elif args.experiment == "usps_with_0.16_noise":
            experiments.openml_experiment("usps_with_0.16_noise", t_const=30)
        elif args.experiment == "usps_with_0.18_noise":
            experiments.openml_experiment("usps_with_0.18_noise", t_const=30)

        elif args.experiment == "4C_with_0.1_noise":
            experiments.openml_experiment("4C_with_0.1_noise", t_const=15)
        elif args.experiment == "4C_with_0.02_noise":
            experiments.openml_experiment("4C_with_0.02_noise", t_const=15)
        elif args.experiment == "4C_with_0.2_noise":
            experiments.openml_experiment("4C_with_0.2_noise", t_const=15)
        elif args.experiment == "4C_with_0.04_noise":
            experiments.openml_experiment("4C_with_0.04_noise", t_const=15)
        elif args.experiment == "4C_with_0.06_noise":
            experiments.openml_experiment("4C_with_0.06_noise", t_const=15)
        elif args.experiment == "4C_with_0.08_noise":
            experiments.openml_experiment("4C_with_0.08_noise", t_const=15)
        elif args.experiment == "4C_with_0.12_noise":
            experiments.openml_experiment("4C_with_0.12_noise", t_const=15)
        elif args.experiment == "4C_with_0.14_noise":
            experiments.openml_experiment("4C_with_0.14_noise", t_const=15)
        elif args.experiment == "4C_with_0.16_noise":
            experiments.openml_experiment("4C_with_0.16_noise", t_const=15)
        elif args.experiment == "4C_with_0.18_noise":
            experiments.openml_experiment("4C_with_0.18_noise", t_const=15)

        elif args.experiment == "pendigits_with_0.1_noise":
            experiments.openml_experiment("pendigits_with_0.1_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.02_noise":
            experiments.openml_experiment("pendigits_with_0.02_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.2_noise":
            experiments.openml_experiment("pendigits_with_0.2_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.04_noise":
            experiments.openml_experiment("pendigits_with_0.04_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.06_noise":
            experiments.openml_experiment("pendigits_with_0.06_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.08_noise":
            experiments.openml_experiment("pendigits_with_0.08_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.12_noise":
            experiments.openml_experiment("pendigits_with_0.12_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.14_noise":
            experiments.openml_experiment("pendigits_with_0.14_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.16_noise":
            experiments.openml_experiment("pendigits_with_0.16_noise", t_const=30)
        elif args.experiment == "pendigits_with_0.18_noise":
            experiments.openml_experiment("pendigits_with_0.18_noise", t_const=30)

        else:
            print("Invalid experiment specified.")


if __name__ == "__main__":
    main()
