import matplotlib.pyplot as plt

nama = ["Andi", "Budi", "Caca", "Deni", "Euis"]
nilai = [85, 90, 78, 82, 88]

plt.plot(nama, nilai, marker="o", linestyle="-", color="skyblue")

plt.figtext(
    0.02,
    1,
    "Riby Imanuel \nA11.22.13969 \nA11.4419",
    ha="left",
    va="top",
    fontsize=10,
    color="black",
)

plt.title("Visualisasi Data Nilai Siswa")
plt.xlabel("Nama Siswa")
plt.ylabel("Nilai")

plt.grid(True)

plt.show()
