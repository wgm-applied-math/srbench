import jessaminescikitlearn.srbench.regressor as JR

est = JR.est
model = JR.model
hyper_params = [
    {
        "op_inventory": [
            "Polynomial",
            "RationalFunction",
            ],
    },
    {
        "p_take_better": [
            0.6,
            0.7,
        ],
    },
]
