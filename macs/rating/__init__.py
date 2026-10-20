"""SC4 human validation: the blind paired-comparison rating study and the independent AI judge.

pairs.py     build the 60 pairs of correct solutions from the traces (blind, seeded)
forms.py     export the rater booklet and the Microsoft Form text; import the Form's responses
analysis.py  Bradley-Terry ranking, Cohen's / Fleiss' kappa, Spearman's rho against novelty
judge.py     the independent AI judge (DeepSeek), asked each pair in both orders

Driven by scripts/rating_study.py and the dashboard's Rating study tab.
"""
