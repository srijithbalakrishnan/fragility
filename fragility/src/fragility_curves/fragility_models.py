import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


class FragilityModel:
    def __init__(self, name):
        """Initiates the fragility model with a name

        :param name: name of the fragility model
        :type name: str
        """
        self.name = name
        self.fragility_curves = dict()

    def set_applicable_compons(self, applicable_compons):
        """Sets the prefixes of components types for which the fragility model is applicable

        :param applicable_compons: The list of prefixes of applicable components types
        :type applicable_compons: list
        """
        self.applicable_compons = applicable_compons

    def set_fragility_curves(
        self, compon_prefix, imt, list_of_states, list_of_median, list_of_beta
    ):
        """Sets the fragility curves for a component type

        :param compon_prefix: The prefix of the component type
        :type compon_prefix: str
        :param imt: The intensity measure type
        :type imt: str
        :param list_of_states: The list of states
        :type list_of_states: list
        :param list_of_median: The list of median values of the fragility curves in the order of states
        :type list_of_median: list
        :param list_of_beta: The list of beta values of the fragility curves in the order of states
        :type list_of_beta: list
        """
        self.fragility_curves[compon_prefix] = dict()
        for index, state in enumerate(list_of_states):
            self.fragility_curves[compon_prefix][state] = dict()
            self.fragility_curves[compon_prefix][state]["imt"] = imt
            self.fragility_curves[compon_prefix][state]["median"] = list_of_median[
                index
            ]
            self.fragility_curves[compon_prefix][state]["beta"] = list_of_beta[index]

    def ascertain_damage_probabilities(
        self, component, imt_type, imt_value, plotting=True
    ):
        """Ascertains the damage of a component type

        :param component: The component type
        :type component: str
        :param imt_value: The intensity measure value
        :type imt_value: float
        """
        compon_type = "".join([i for i in component if not i.isdigit()])
        if compon_type in self.fragility_curves.keys():
            state_cdf = []
            for state in self.fragility_curves[compon_type].keys():
                imt = self.fragility_curves[compon_type][state]["imt"]
                if imt == imt_type:
                    median = self.fragility_curves[compon_type][state]["median"]
                    beta = self.fragility_curves[compon_type][state]["beta"]
                    state_cdf.append(
                        self.calculate_state_probability(imt_value, median, beta)
                    )
                else:
                    print(
                        "The available fragility curves are not applicable for the component type"
                    )
            if len(state_cdf) == len(self.fragility_curves[compon_type].keys()):
                remain_prob = 1
                state_probabilities = []
                for cumprob in state_cdf:
                    state_prob = remain_prob - cumprob
                    state_probabilities.append(state_prob)
                    remain_prob -= state_prob

            if plotting:
                imt_list = np.linspace(0.001, imt_value * 2, 100)

                frag_df = pd.DataFrame(columns=["imt", "state", "fragility"])

                for state in self.fragility_curves[compon_type].keys():
                    median = self.fragility_curves[compon_type][state]["median"]
                    stdev = self.fragility_curves[compon_type][state]["beta"]
                    for imt in imt_list:
                        frag_df = frag_df.append(
                            {
                                "imt": imt,
                                "state": state,
                                "fragility": self.calculate_state_probability(
                                    imt, median, stdev
                                ),
                            },
                            ignore_index=True,
                        )
                sns.set_style("ticks")
                sns.set_context("paper", font_scale=1.25)

                fig, ax = plt.subplots(figsize=(7, 4))
                sns.lineplot(x="imt", y="fragility", hue="state", data=frag_df)

                ax.hlines(
                    y=state_cdf,
                    xmin=0,
                    xmax=imt_value,
                    color="grey",
                    linestyle="--",
                )
                ax.vlines(
                    x=imt_value,
                    ymin=0,
                    ymax=max(state_cdf),
                    color="grey",
                    linestyle="--",
                )
                ax.set_ylabel("Damage state probability (cumulative)")
                ax.set_xlabel(f"{imt_type}")

                ax.set_xlim(0, imt_value * 2)
                ax.set_ylim(0, 1)

            return state_probabilities

    @staticmethod
    def calculate_state_probability(imt, median, beta):
        val = np.log(imt / median) / beta
        return norm.cdf(val)
