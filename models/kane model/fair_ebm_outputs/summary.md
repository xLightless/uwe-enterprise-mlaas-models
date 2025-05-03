# Fair Energy-Based Model (EBM) Training Summary

## Best Hyperparameters
- num_bins: 40
- l2_regularization: 0.00014908669250618053
- learning_rate: 0.003106771530342737
- batch_size: 256
- epochs: 125
- fairness_penalty: 0.17188518561240726
- use_interactions: True
- num_interactions: 7

- Number of interactions: 7

## Fairness Configuration
- Sensitive features: ['numeric__Driver Age', 'categorical__Gender_Female', 'categorical__Gender_Male', 'categorical__Gender_Other']
- Fairness penalty: 0.17188518561240726

## Accuracy Metrics
- MSE (original): 163252.2963
- RMSE (original): 404.0449
- MAE (original): 217.1439
- R² (original): 0.7840
- MSE (transformed): 0.0588
- RMSE (transformed): 0.2425
- MAE (transformed): 0.1656
- R² (transformed): 0.8568


## Fairness Metrics
- numeric__Driver Age_disparate_impact: 0.3732
- numeric__Driver Age_stat_parity_diff: 0.1635
- numeric__Driver Age_-1.7498986588146377_mean_error: 0.2075
- numeric__Driver Age_-1.7498986588146377_count: 15.0000
- numeric__Driver Age_-1.6930795454933425_mean_error: 0.1392
- numeric__Driver Age_-1.6930795454933425_count: 17.0000
- numeric__Driver Age_-1.6362604321720478_mean_error: 0.1608
- numeric__Driver Age_-1.6362604321720478_count: 21.0000
- numeric__Driver Age_-1.579441318850753_mean_error: 0.1811
- numeric__Driver Age_-1.579441318850753_count: 17.0000
- numeric__Driver Age_-1.5226222055294585_mean_error: 0.1099
- numeric__Driver Age_-1.5226222055294585_count: 15.0000
- numeric__Driver Age_-1.4658030922081635_mean_error: 0.1341
- numeric__Driver Age_-1.4658030922081635_count: 20.0000
- numeric__Driver Age_-1.4089839788868683_mean_error: 0.1949
- numeric__Driver Age_-1.4089839788868683_count: 18.0000
- numeric__Driver Age_-1.3521648655655738_mean_error: 0.1710
- numeric__Driver Age_-1.3521648655655738_count: 7.0000
- numeric__Driver Age_-1.295345752244279_mean_error: 0.1607
- numeric__Driver Age_-1.295345752244279_count: 15.0000
- numeric__Driver Age_-1.2385266389229843_mean_error: 0.1285
- numeric__Driver Age_-1.2385266389229843_count: 10.0000
- numeric__Driver Age_-1.1817075256016896_mean_error: 0.2335
- numeric__Driver Age_-1.1817075256016896_count: 15.0000
- numeric__Driver Age_-1.1248884122803946_mean_error: 0.1821
- numeric__Driver Age_-1.1248884122803946_count: 18.0000
- numeric__Driver Age_-1.0680692989591_mean_error: 0.1573
- numeric__Driver Age_-1.0680692989591_count: 13.0000
- numeric__Driver Age_-1.011250185637805_mean_error: 0.1501
- numeric__Driver Age_-1.011250185637805_count: 20.0000
- numeric__Driver Age_-0.9544310723165104_mean_error: 0.1868
- numeric__Driver Age_-0.9544310723165104_count: 11.0000
- numeric__Driver Age_-0.8976119589952155_mean_error: 0.1824
- numeric__Driver Age_-0.8976119589952155_count: 19.0000
- numeric__Driver Age_-0.8407928456739208_mean_error: 0.1949
- numeric__Driver Age_-0.8407928456739208_count: 9.0000
- numeric__Driver Age_-0.783973732352626_mean_error: 0.1509
- numeric__Driver Age_-0.783973732352626_count: 18.0000
- numeric__Driver Age_-0.7271546190313312_mean_error: 0.2148
- numeric__Driver Age_-0.7271546190313312_count: 5.0000
- numeric__Driver Age_-0.6703355057100364_mean_error: 0.1709
- numeric__Driver Age_-0.6703355057100364_count: 10.0000
- numeric__Driver Age_-0.6135163923887416_mean_error: 0.1373
- numeric__Driver Age_-0.6135163923887416_count: 18.0000
- numeric__Driver Age_-0.5566972790674468_mean_error: 0.2580
- numeric__Driver Age_-0.5566972790674468_count: 16.0000
- numeric__Driver Age_-0.499878165746152_mean_error: 0.1459
- numeric__Driver Age_-0.499878165746152_count: 19.0000
- numeric__Driver Age_-0.4430590524248572_mean_error: 0.1591
- numeric__Driver Age_-0.4430590524248572_count: 17.0000
- numeric__Driver Age_-0.3862399391035624_mean_error: 0.1692
- numeric__Driver Age_-0.3862399391035624_count: 19.0000
- numeric__Driver Age_-0.3294208257822676_mean_error: 0.1685
- numeric__Driver Age_-0.3294208257822676_count: 20.0000
- numeric__Driver Age_-0.2726017124609728_mean_error: 0.2329
- numeric__Driver Age_-0.2726017124609728_count: 14.0000
- numeric__Driver Age_-0.2157825991396781_mean_error: 0.1589
- numeric__Driver Age_-0.2157825991396781_count: 14.0000
- numeric__Driver Age_-0.1589634858183833_mean_error: 0.1309
- numeric__Driver Age_-0.1589634858183833_count: 16.0000
- numeric__Driver Age_-0.1021443724970885_mean_error: 0.1438
- numeric__Driver Age_-0.1021443724970885_count: 14.0000
- numeric__Driver Age_-0.0453252591757937_mean_error: 0.2609
- numeric__Driver Age_-0.0453252591757937_count: 10.0000
- numeric__Driver Age_0.011493854145501_mean_error: 0.2008
- numeric__Driver Age_0.011493854145501_count: 36.0000
- numeric__Driver Age_0.0683129674667958_mean_error: 0.1869
- numeric__Driver Age_0.0683129674667958_count: 14.0000
- numeric__Driver Age_0.1251320807880906_mean_error: 0.1318
- numeric__Driver Age_0.1251320807880906_count: 13.0000
- numeric__Driver Age_0.1819511941093854_mean_error: 0.1307
- numeric__Driver Age_0.1819511941093854_count: 21.0000
- numeric__Driver Age_0.2387703074306802_mean_error: 0.1349
- numeric__Driver Age_0.2387703074306802_count: 15.0000
- numeric__Driver Age_0.295589420751975_mean_error: 0.1665
- numeric__Driver Age_0.295589420751975_count: 10.0000
- numeric__Driver Age_0.3524085340732698_mean_error: 0.1001
- numeric__Driver Age_0.3524085340732698_count: 20.0000
- numeric__Driver Age_0.4092276473945646_mean_error: 0.1645
- numeric__Driver Age_0.4092276473945646_count: 18.0000
- numeric__Driver Age_0.4660467607158593_mean_error: 0.1739
- numeric__Driver Age_0.4660467607158593_count: 15.0000
- numeric__Driver Age_0.5228658740371541_mean_error: 0.2504
- numeric__Driver Age_0.5228658740371541_count: 14.0000
- numeric__Driver Age_0.5796849873584489_mean_error: 0.1246
- numeric__Driver Age_0.5796849873584489_count: 19.0000
- numeric__Driver Age_0.6365041006797437_mean_error: 0.0974
- numeric__Driver Age_0.6365041006797437_count: 19.0000
- numeric__Driver Age_0.6933232140010385_mean_error: 0.1036
- numeric__Driver Age_0.6933232140010385_count: 16.0000
- numeric__Driver Age_0.7501423273223333_mean_error: 0.2444
- numeric__Driver Age_0.7501423273223333_count: 8.0000
- numeric__Driver Age_0.8069614406436281_mean_error: 0.1675
- numeric__Driver Age_0.8069614406436281_count: 19.0000
- numeric__Driver Age_0.8637805539649228_mean_error: 0.2438
- numeric__Driver Age_0.8637805539649228_count: 16.0000
- numeric__Driver Age_0.9205996672862176_mean_error: 0.1659
- numeric__Driver Age_0.9205996672862176_count: 23.0000
- numeric__Driver Age_0.9774187806075124_mean_error: 0.1183
- numeric__Driver Age_0.9774187806075124_count: 13.0000
- numeric__Driver Age_1.0342378939288073_mean_error: 0.2039
- numeric__Driver Age_1.0342378939288073_count: 23.0000
- numeric__Driver Age_1.091057007250102_mean_error: 0.1891
- numeric__Driver Age_1.091057007250102_count: 20.0000
- numeric__Driver Age_1.1478761205713968_mean_error: 0.1625
- numeric__Driver Age_1.1478761205713968_count: 11.0000
- numeric__Driver Age_1.2046952338926915_mean_error: 0.1373
- numeric__Driver Age_1.2046952338926915_count: 19.0000
- numeric__Driver Age_1.2615143472139865_mean_error: 0.1796
- numeric__Driver Age_1.2615143472139865_count: 15.0000
- numeric__Driver Age_1.3183334605352812_mean_error: 0.1341
- numeric__Driver Age_1.3183334605352812_count: 8.0000
- numeric__Driver Age_1.375152573856576_mean_error: 0.1597
- numeric__Driver Age_1.375152573856576_count: 14.0000
- numeric__Driver Age_1.4319716871778707_mean_error: 0.1685
- numeric__Driver Age_1.4319716871778707_count: 11.0000
- numeric__Driver Age_1.4887908004991657_mean_error: 0.1522
- numeric__Driver Age_1.4887908004991657_count: 18.0000
- numeric__Driver Age_1.5456099138204602_mean_error: 0.2302
- numeric__Driver Age_1.5456099138204602_count: 16.0000
- numeric__Driver Age_1.6024290271417552_mean_error: 0.0977
- numeric__Driver Age_1.6024290271417552_count: 16.0000
- numeric__Driver Age_1.65924814046305_mean_error: 0.1327
- numeric__Driver Age_1.65924814046305_count: 12.0000
- numeric__Driver Age_1.7160672537843449_mean_error: 0.1407
- numeric__Driver Age_1.7160672537843449_count: 17.0000
- categorical__Gender_Female_disparate_impact: 0.9553
- categorical__Gender_Female_stat_parity_diff: 0.0075
- categorical__Gender_Female_0.0_mean_error: 0.1682
- categorical__Gender_Female_0.0_count: 650.0000
- categorical__Gender_Female_1.0_mean_error: 0.1606
- categorical__Gender_Female_1.0_count: 329.0000
- categorical__Gender_Male_disparate_impact: 0.9971
- categorical__Gender_Male_stat_parity_diff: 0.0005
- categorical__Gender_Male_0.0_mean_error: 0.1655
- categorical__Gender_Male_0.0_count: 639.0000
- categorical__Gender_Male_1.0_mean_error: 0.1659
- categorical__Gender_Male_1.0_count: 340.0000
- categorical__Gender_Other_disparate_impact: 0.9575
- categorical__Gender_Other_stat_parity_diff: 0.0073
- categorical__Gender_Other_0.0_mean_error: 0.1633
- categorical__Gender_Other_0.0_count: 669.0000
- categorical__Gender_Other_1.0_mean_error: 0.1706
- categorical__Gender_Other_1.0_count: 310.0000
- overall_fairness: 0.8208


## Top 10 Aggregated Feature Importances
- SpecialHealthExpenses: 0.0000
- SpecialOverage: 0.0000
- SpecialAdditionalInjury: 0.0000
- SpecialEarningsLoss: 0.0000
- SpecialUsageLoss: 0.0000
- SpecialMedications: 0.0000
- SpecialAssetDamage: 0.0000
- SpecialRehabilitation: 0.0000
- SpecialFixes: 0.0000
- SpecialLoanerVehicle: 0.0000


## Model Summary
This model uses group-wise regularization to ensure that sensitive attributes
have minimal or equal contributions to predictions, promoting fairness.

Feature importances are aggregated from one-hot encoded features to their original categorical variables.

Model artifacts saved in: fair_ebm_outputs