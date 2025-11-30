import os


def batch_rename_files(klasor, prefix):
    os.chdir(klasor)
    dosyalar = os.listdir()
    for i in range(len(dosyalar)):
        eski_isim = dosyalar[i]
        yeni_isim = f"{prefix}_{eski_isim}"
        os.rename(eski_isim, yeni_isim)
    print("Dosyalar başarıyla yeniden adlandırıldı.")


# Örnek kullanım
if __name__ == "__main__":
    hedef_klasor = "General Examples/test_rename_icin"
    # windowapth=r"falanca\filanca\xyz"
    yeni_ön_ek = input("Yeni ön eki girin: ")
    batch_rename_files(hedef_klasor, yeni_ön_ek)
