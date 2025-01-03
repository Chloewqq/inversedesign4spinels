import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import subprocess
import re

class FeatureEngineer:
    def __init__(self, formulas: pd.Series, y: pd.DataFrame):
        self.formulas = formulas
        self.y = y

    def get_magpie_plus_features(self, formulas):
        input_file = "inputFile"
        output_file = "out.csv"
        formulas.to_csv(input_file, index=False, header=False, sep='\n')

        command = ['java', '-jar', 'MaterialDescriptors.jar', input_file, '0', output_file]
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("Java program error output:", e.output.decode())
            raise

        output_str = output.decode()
        comp_number = re.findall(r'\d+', output_str)[0]
        print(f'成功生成了{comp_number}个化学式的特征！')

        df = pd.read_csv(output_file)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df

    def run_featurization(self):
        temp_df = pd.DataFrame({'formula': self.formulas})

        def parse_formula(spinels, formula):
            C = ['O', 'S', 'Se', 'Te','Cl','F']
            elements = re.findall('([A-Z][a-z]*)(\d*\.*\d*)|($(?:[A-Z][a-z]*\d*\.*\d*)+$(\d*\.*\d*))', formula)

            spinels.loc[len(spinels)] = [formula, "", "", "", "", "", "", 0, 0, 0, 0, 0, 0, 0]
            idx = len(spinels) - 1
            A_count, B_count, C_count = 0, 0, 0
            A_total, B_total, C_total = 0, 0, 0

            for element, coefficient, group, group_coefficient in elements:
                if group:
                    group_elements = re.findall('([A-Z][a-z]*)(\d*\.*\d*)', group)
                    group_coefficient = float(group_coefficient) if group_coefficient else 1
                    for element, coefficient in group_elements:
                        coefficient = float(coefficient) if coefficient else 1
                        coefficient *= group_coefficient
                        if element not in C:
                            if A_total + coefficient <= 1:
                                if A_count < 2:
                                    spinels.at[idx, 'A'+str(A_count+1)] = element
                                    spinels.at[idx, 'A'+str(A_count+1)+'_v'] = coefficient
                                    A_count += 1
                                A_total += coefficient
                            elif B_total + coefficient <= 2:
                                if B_count < 2:
                                    spinels.at[idx, 'B'+str(B_count+1)] = element
                                    spinels.at[idx, 'B'+str(B_count+1)+'_v'] = coefficient
                                    B_count += 1
                                B_total += coefficient
                        else:
                            if C_count < 2:
                                spinels.at[idx, 'C'+str(C_count+1)] = element
                                spinels.at[idx, 'C'+str(C_count+1)+'_v'] = coefficient
                                C_count += 1
                            C_total += coefficient
                else:
                    coefficient = float(coefficient) if coefficient else 1
                    if element not in C:
                        if A_total + coefficient <= 1:
                            if A_count < 2:
                                spinels.at[idx, 'A'+str(A_count+1)] = element
                                spinels.at[idx, 'A'+str(A_count+1)+'_v'] = coefficient
                                A_count += 1
                            A_total += coefficient
                        elif B_total + coefficient <= 2:
                            if B_count < 2:
                                spinels.at[idx, 'B'+str(B_count+1)] = element
                                spinels.at[idx, 'B'+str(B_count+1)+'_v'] = coefficient
                                B_count += 1
                            B_total += coefficient
                    else:
                        if C_count < 2:
                            spinels.at[idx, 'C'+str(C_count+1)] = element
                            spinels.at[idx, 'C'+str(C_count+1)+'_v'] = coefficient
                            C_count += 1
                        C_total += coefficient

            if spinels.at[idx, 'A2'] == "":
                spinels.at[idx, 'A2'] = spinels.at[idx, 'A1']
                spinels.at[idx, 'A2_v'] = 0
            if spinels.at[idx, 'B2'] == "":
                spinels.at[idx, 'B2'] = spinels.at[idx, 'B1']
                spinels.at[idx, 'B2_v'] = 0
            if spinels.at[idx, 'C2'] == "":
                spinels.at[idx, 'C2'] = spinels.at[idx, 'C1']
                spinels.at[idx, 'C2_v'] = 0

            return spinels

        spinels = pd.DataFrame(columns=['formula', 'A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'A1_v', 'A2_v', 'B1_v', 'B2_v', 'C1_v', 'C2_v', 'extra'])
        for formula in temp_df['formula']:
            spinels = parse_formula(spinels, formula)
        
        from Featurizor3 import Featurizor
        zero_rows = spinels_data[(spinels_data['C1_v'] == 0) & (spinels_data['C2_v'] == 0)]
        if not zero_rows.empty:
            print("发现 C1_v 和 C2_v 都为零的行:")
            print(zero_rows[['formula', 'C1', 'C2', 'C1_v', 'C2_v']])
            print(f"总共有 {len(zero_rows)} 行 C1_v 和 C2_v 都为零")

        remaining_formulas = spinels_data['formula']
        mp_f = self.get_magpie_plus_features(remaining_formulas)

        spinels_data = pd.concat([spinels_data.reset_index(drop=True), mp_f.reset_index(drop=True)], axis=1)
        X = spinels_data.iloc[:, 1:]

        return X, spinels_data['formula']

    def remove_zero_columns(self, X, threshold=0.85, remove_nan_cols=True):
        zero_ratio = (X == 0).sum() / len(X)
        cols_to_drop = zero_ratio[zero_ratio > threshold].index
        
        if remove_nan_cols:
            nan_cols = X.columns[X.isna().any()].tolist()
            cols_to_drop = list(set(cols_to_drop) | set(nan_cols))
        
        X = X.drop(cols_to_drop, axis=1)
        return X

    def remove_collinear_features(self, X, y, threshold=0.8):
        def remove_correlated_features(x, y_values, threshold):
            corr_matrix = x.corr()
            drop_cols = []
            for i in range(len(corr_matrix.columns) - 1):
                for j in range(i):
                    item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                    if abs(item.values) >= threshold:
                        col = item.columns[0]
                        row = item.index[0]
                        if not pd.isnull(y_values).any():
                            x1 = x[col]
                            x2 = x[row]
                            corr1, _ = spearmanr(x1, y_values)
                            corr2, _ = spearmanr(x2, y_values)
                            if abs(corr1) < abs(corr2):
                                drop_cols.append(col)
                            else:
                                drop_cols.append(row)

            drops = set(drop_cols)
            x = x.drop(columns=drops)
            return x

        y_values = y['bandgap'].dropna()
        drop_index = y[y['bandgap'].isnull()].index
    
        X = X.drop(drop_index)
        return remove_correlated_features(X, y_values, threshold)

    def targets_process(self, y: pd.DataFrame):
        # # HSE Correct bandgap

        from modify import Modifier
        modifier = Modifier()
        spinel_test_hse = modifier.modify(y['formula'].values,y['bandgap'].values)
        y['bandgap'] = spinel_test_hse['hse']
        y['is_gapd'] = y['is_gapd'].apply(lambda x: 1 if x is True else 0 if x is False else np.nan)
        y['ehull'] = y['ehull'].apply(lambda x: 1 if (x <= 0.025 and x is not np.nan) else 0 if (x is not np.nan) else np.nan)
        y['bandgap'] = y['bandgap'].apply(lambda x: 1 if (0.9 <= x <= 2 and x is not np.nan) else 0 if (x is not np.nan) else np.nan)
        return y

    def process_features(self):
        X, formulas = self.run_featurization()
        print("X:",X.shape)
        
        self.y = self.y.loc[formulas.index]
        print("y:",self.y.shape)
        X = X.loc[formulas.index]  # Ensure X and y have the same index
        print("X_formula:",X.shape)
        
        # 调用修改后的方法去除含有 0 的列和含有 NaN 的列
        X = self.remove_zero_columns(X, remove_nan_cols=True)
        print("X_remove_zero_and_nan:",X.shape)
        
        X = self.remove_collinear_features(X, self.y)
        print("X:",X.shape)
        
        y = self.targets_process(self.y)
        return X, y

# sample
# engineer = FeatureEngineer(df['formula'], y)
# X, y = engineer.process_features()