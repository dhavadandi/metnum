import pandas as pd
import numpy as np

# FUNGSI: LAGRANGE INTERPOLATION

def lagrange_interpolation(x_values, y_values, x_target):
    n = len(x_values)
    L = np.zeros(n)

    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x_target - x_values[j]) / (x_values[i] - x_values[j])
        L[i] = Li

    # f(x) = Σ Li * f(xi)
    f_est = np.sum(L * y_values)
    return L, f_est

# STUDY CASE 1 — DATA KELAHIRAN NEGARA X

x1 = np.array([2020, 2021, 2023])      
y1 = np.array([4.69, 4.67, 4.62])        
x_target1 = 2022                         
exact1 = 4.65                            

L1, f2 = lagrange_interpolation(x1, y1, x_target1)

df1 = pd.DataFrame({
    "i"      : [0, 1, 2],
    "xi"     : x1,
    "f(xi)"  : y1,
    "Li(x)"  : L1.round(3),
    "Li(x)f(xi)" : (L1 * y1).round(3)
})

error1 = abs(exact1 - f2) / exact1 * 100

print("\n       STUDY CASE 2 \n")
print(df1)
print(f"\nf₂(2022) = {f2:.3f}")
print(f"Exact    = {exact1}")
print(f"Error    = {error1:.3f}%\n")

print("Kesimpulan:")
print(f"Dengan interpolasi Lagrange, estimasi angka kelahiran tahun 2022 adalah {f2:.3f} ribu.\n")

# STUDY CASE 2 — DATA KEUNTUNGAN USAHA FOTOKOPI

x2 = np.array([3, 4, 6, 7])    
y2 = np.array([56491, 42449, 48762, 63718])
x_target2 = 5                  
exact2 = 39601                 

L2, f3 = lagrange_interpolation(x2, y2, x_target2)

df2 = pd.DataFrame({
    "i"      : [0, 1, 2, 3],
    "xi"     : x2,
    "f(xi)"  : y2,
    "Li(x)"  : L2.round(3),
    "Li(x)f(xi)" : (L2 * y2).round(2)
})

error2 = abs(exact2 - f3) / exact2 * 100

print("\n       STUDY CASE 2 \n")
print(df2)
print(f"\nf₃(Mei) = {f3:,.2f}")
print(f"Exact   = {exact2:,}")
print(f"Error   = {error2:.3f}%\n")

print("Kesimpulan:")
print(f"Dengan interpolasi Lagrange, estimasi keuntungan bulan Mei adalah Rp {f3:,.2f}.\n")
