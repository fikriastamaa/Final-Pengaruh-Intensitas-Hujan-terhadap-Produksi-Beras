import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Fungsi untuk penggabungan data
def preprocess_data(curah_hujan, produksi_padi):
    data_hujan_long = pd.melt(curah_hujan, id_vars=['Bulan'], var_name='Tahun', value_name='Intensitas_Hujan')
    data_hujan_long['Tahun'] = data_hujan_long['Tahun'].str.replace('Intensitas_', '')
    
    data_padi_long = pd.melt(produksi_padi, id_vars=['Bulan'], var_name='Tahun', value_name='Jumlah_Produksi_Beras')
    data_padi_long['Tahun'] = data_padi_long['Tahun'].str.replace('Jumlah_Produksi_Beras_', '')

    data_merged = pd.merge(data_hujan_long, data_padi_long, on=['Bulan', 'Tahun'])
    bulan_mapping = {"Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
                     "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12}
    data_merged['Bulan_Angka'] = data_merged['Bulan'].map(bulan_mapping)
    return data_merged

# Navigasi Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title="Dashboard Navigation",
        options=["Upload Data", "Eksplorasi Data", "Model dan Evaluasi", "Prediksi", "About"],
        icons=["cloud-upload", "bar-chart", "gear", "graph-up-arrow", "info-circle"],
        menu_icon="cast",
        default_index=0
    )

if selected == "Upload Data":
    st.title("Upload Dataset")
    st.write("Unggah file dataset curah hujan dan produksi padi dalam format CSV.")

    uploaded_hujan = st.file_uploader("Unggah dataset curah hujan:", type=["csv"])
    uploaded_padi = st.file_uploader("Unggah dataset produksi padi:", type=["csv"])

    if uploaded_hujan and uploaded_padi:
        curah_hujan = pd.read_csv(uploaded_hujan)
        produksi_padi = pd.read_csv(uploaded_padi)
        st.session_state['curah_hujan'] = curah_hujan
        st.session_state['produksi_padi'] = produksi_padi
        st.success("Dataset berhasil diunggah!")

elif selected == "Eksplorasi Data":
    st.title("Eksplorasi Data")
    if 'curah_hujan' in st.session_state and 'produksi_padi' in st.session_state:
        curah_hujan = st.session_state['curah_hujan']
        produksi_padi = st.session_state['produksi_padi']
        data_merged = preprocess_data(curah_hujan, produksi_padi)

        st.write("### Data Curah Hujan")
        st.write(curah_hujan)

        st.write("### Data Produksi Padi")
        st.write(produksi_padi)

        st.write("### Data Gabungan")
        st.write(data_merged)

        # Visualisasi Hubungan Variabel
        st.write("### Hubungan Curah Hujan dan Produksi Padi")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=data_merged, x='Intensitas_Hujan', y='Jumlah_Produksi_Beras', hue='Tahun', ax=ax)
        plt.title("Hubungan Curah Hujan dan Produksi Padi")
        st.pyplot(fig)

        # Line plot untuk tren bulanan curah hujan
        st.write("### Tren Bulanan Curah Hujan")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data_merged, x='Bulan', y='Intensitas_Hujan', hue='Tahun', marker='o', palette='tab10', ax=ax)
        plt.title('Tren Bulanan Curah Hujan', fontsize=14)
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Intensitas Hujan (mm³)', fontsize=12)
        plt.legend(title='Tahun')
        plt.grid()
        st.pyplot(fig)

        # Line plot untuk tren bulanan Produksi Beras
        st.write("### Tren Bulanan Produksi Beras")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data_merged, x='Bulan', y='Jumlah_Produksi_Beras', hue='Tahun', marker='o', palette='tab10', ax=ax)
        plt.title('Tren Bulanan Produksi Beras', fontsize=14)
        plt.xlabel('Bulan', fontsize=12)
        plt.ylabel('Produksi Beras (ton)', fontsize=12)
        plt.legend(title='Tahun')
        plt.grid()
        st.pyplot(fig)

        # Grafik Curah Hujan dan Produksi Padi
        st.write("### Grafik Curah Hujan dan Produksi Padi")
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax2 = ax1.twinx()
        sns.lineplot(data=data_merged, x='Bulan', y='Intensitas_Hujan', hue='Tahun', marker='o', ax=ax1, palette='Blues')
        sns.lineplot(data=data_merged, x='Bulan', y='Jumlah_Produksi_Beras', hue='Tahun', marker='s', ax=ax2, palette='Greens')

        ax1.set_title("Curah Hujan dan Produksi Padi per Bulan", fontsize=14)
        ax1.set_xlabel('Bulan', fontsize=12)
        ax1.set_ylabel('Intensitas Hujan (mm³)', fontsize=12, color='blue')
        ax2.set_ylabel('Jumlah Produksi Padi (ton)', fontsize=12, color='green')

        plt.grid()
        st.pyplot(fig)

        # Pie Chart untuk Total Curah Hujan dan Produksi Padi per Tahun
        st.write("### Pie Chart: Total Curah Hujan dan Produksi Padi per Tahun")

        # Total Curah Hujan per Tahun
        total_hujan = data_merged.groupby('Tahun')['Intensitas_Hujan'].sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(total_hujan, labels=total_hujan.index, autopct='%1.1f%%', colors=sns.color_palette('Blues'), startangle=90)
        ax.set_title("Distribusi Total Curah Hujan per Tahun")
        st.pyplot(fig)

        # Total Produksi Padi per Tahun
        total_padi = data_merged.groupby('Tahun')['Jumlah_Produksi_Beras'].sum()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(total_padi, labels=total_padi.index, autopct='%1.1f%%', colors=sns.color_palette('Greens'), startangle=90)
        ax.set_title("Distribusi Total Produksi Padi per Tahun")
        st.pyplot(fig)

        # Heatmap korelasi variabel numerik
        st.write("### Heatmap Korelasi Variabel Numerik")
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = data_merged[['Intensitas_Hujan', 'Jumlah_Produksi_Beras']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title('Heatmap Korelasi Variabel Numerik', fontsize=14)
        st.pyplot(fig)

    else:
        st.warning("Harap unggah dataset terlebih dahulu di halaman 'Upload Data'.")

elif selected == "Model dan Evaluasi":
    st.title("MODEL DAN EVALUASI")
    
    if 'curah_hujan' in st.session_state and 'produksi_padi' in st.session_state:
        curah_hujan = st.session_state['curah_hujan']
        produksi_padi = st.session_state['produksi_padi']
        data_merged = preprocess_data(curah_hujan, produksi_padi)

        # Split Data
        X = data_merged[['Intensitas_Hujan', 'Bulan_Angka']]
        y = data_merged['Jumlah_Produksi_Beras']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Random Forest
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi Model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Deskripsi Model")
        st.write("""
        **Random Forest Regressor** digunakan untuk memprediksi jumlah produksi padi berdasarkan data intensitas hujan 
        dan waktu (bulan dalam angka). Data pelatihan (80%) digunakan untuk melatih model, sedangkan data pengujian (20%) 
        digunakan untuk mengevaluasi kinerjanya. Model ini memanfaatkan pendekatan ensemble untuk meningkatkan akurasi
        prediksi dibanding metode regresi linear konvensional.
        """)
        st.write("""
        Berikut adalah formula sederhana yang dapat diterapkan dalam Random Forest (dibentuk oleh algoritma ensemble):
        
        **Prediksi** = Rata-rata dari nilai-nilai hasil regresi dari banyak pohon keputusan yang dibuat.
        """)

        # Hasil Model
        st.write("### Evaluasi Model")
        st.write(f"**Model Accuracy (R-Squared):** {r2}")
        st.write(f"**Mean Squared Error (MSE):** {mse}")

        st.write("#### Model Evaluation Overview")
        st.write(f"""
        1. **Model Accuracy (R²)** memberikan skor sebesar **{r2}**, menunjukkan bahwa model dapat menjelaskan sekitar 
        **{r2 * 100:.1f}%** dari variasi jumlah produksi padi dari data pengujian.
        2. **MSE** mencatat error rata-rata pada data pengujian sebesar **{mse:.2f}** ton.
        """)

        # Visualisasi Model Prediksi vs Aktual
        st.write("### Visualisasi Prediksi vs Aktual")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.7, label="Prediksi")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2, label="Perfect Fit")
        plt.title("Prediksi vs Data Aktual", fontsize=14)
        plt.xlabel("Produksi Padi Aktual (ton)", fontsize=12)
        plt.ylabel("Produksi Padi Prediksi (ton)", fontsize=12)
        plt.legend()
        plt.grid()
        st.pyplot(fig)

        # Visualisasi Intensitas Hujan vs Produksi Beras
        st.write("### Visualisasi Intensitas Hujan vs Produksi Aktual dan Prediksi")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X_test['Intensitas_Hujan'], y_test, alpha=0.7, label="Aktual", c="blue")
        ax.scatter(X_test['Intensitas_Hujan'], y_pred, alpha=0.7, label="Prediksi", c="orange")
        plt.title("Intensitas Hujan vs Produksi Padi", fontsize=14)
        plt.xlabel("Intensitas Hujan (mm³)", fontsize=12)
        plt.ylabel("Produksi Padi (ton)", fontsize=12)
        plt.legend()
        plt.grid()
        st.pyplot(fig)

    else:
        st.warning("Harap unggah dataset terlebih dahulu di halaman 'Upload Data'.")

elif selected == "Prediksi":
    st.title("Prediksi Produksi Padi")
    st.write("Masukkan nilai untuk melakukan prediksi produksi padi berdasarkan intensitas hujan dan bulan.")

    if 'curah_hujan' in st.session_state and 'produksi_padi' in st.session_state:
        curah_hujan = st.session_state['curah_hujan']
        produksi_padi = st.session_state['produksi_padi']
        data_merged = preprocess_data(curah_hujan, produksi_padi)

        # Model Training
        X = data_merged[['Intensitas_Hujan', 'Bulan_Angka']]
        y = data_merged['Jumlah_Produksi_Beras']
        model = RandomForestRegressor(random_state=42, n_estimators=100)
        model.fit(X, y)

        # Input User
        intensitas_hujan = st.number_input("Masukkan Intensitas Hujan (mm³):", min_value=0.0, step=1.0)
        bulan = st.selectbox("Pilih Bulan:", list(range(1, 13)))

        if st.button("Prediksi"):
            # Prediksi
            prediksi = model.predict([[intensitas_hujan, bulan]])
            st.write(f"### Prediksi Produksi Padi: {prediksi[0]:.2f} ton")
    else:
        st.warning("Harap unggah dataset terlebih dahulu di halaman 'Upload Data'.")

elif selected == "About":
    st.title("About")
    st.write("""
    **Aplikasi Prediksi Produksi Padi** ini dibuat oleh :
    - **Nama:** [Fikri Astama Putra] 123220108
    - **Nama:** [Anugrah Mohammad Fadillah] 123220062
    - **Nama:** [Serrano Mantara] 123220003
    """)

    st.markdown("---")

    if 'curah_hujan' in st.session_state and 'produksi_padi' in st.session_state:
        st.write("# Business Understanding & Analytic Approach")

        st.write("""
        Wilayah Daerah Istimewa Yogyakarta (DIY) memiliki peran penting dalam sektor agrikultur, khususnya dalam produksi padi. 
        Namun, salah satu tantangan utama yang dihadapi adalah **ketidakpastian intensitas hujan akibat perubahan iklim.** 
        Intensitas hujan yang tidak stabil sering kali mempengaruhi hasil panen, mulai dari risiko kekeringan yang menghambat 
        pertumbuhan padi hingga genangan air berlebihan yang merusak lahan pertanian.

        Proyek ini bertujuan untuk:
        - Memahami pengaruh **intensitas hujan** terhadap hasil produksi padi di DIY.
        - Melakukan **prediksi pola curah hujan** untuk membantu perencanaan agrikultur yang lebih efektif.
        - Membantu **petani dan pemangku kebijakan** memitigasi risiko iklim sekaligus meningkatkan produktivitas lahan pertanian.
        """)

        st.write("""
        ### Tujuan dan Pendekatan
        Untuk mencapai tujuan tersebut, berikut adalah langkah-langkah utama dalam proyek ini:
        1. **Mendefinisikan Business Understanding**:
           - Menjelaskan tantangan ketidakpastian curah hujan di Yogyakarta dan dampaknya terhadap produktivitas pertanian, khususnya hasil panen padi.
        2. **Menentukan Analytic Approach**:
           - Menggunakan pendekatan **Predictive** untuk menganalisis hubungan antara intensitas hujan dengan hasil panen padi, serta memprediksi pola curah hujan di masa depan.
        3. **Mengumpulkan dan Menganalisis Data**:
           - Data yang digunakan meliputi intensitas hujan (mm³) per bulan, hasil produksi padi (ton) per kabupaten/kota, serta data pendukung lain yang relevan.
        4. **Menentukan Jenis Model yang Digunakan**:
           - **Random Forest Regression**, karena efektif dalam menganalisis data multivariat dan menangani hubungan non-linear antar variabel.
        5. **Menentukan Algoritma yang Digunakan**:
           - Random Forest digunakan karena kemampuannya yang tinggi dalam menangani outlier, korelasi antar variabel, dan memberikan interpretasi melalui fitur penting (feature importance).
        6. **Mendeskripsikan Hasil yang Dicapai**:
           - Menampilkan wawasan mendalam tentang hubungan intensitas hujan dengan hasil panen, serta prediksi curah hujan bulanan di wilayah Yogyakarta untuk mendukung perencanaan agrikultur.
        """)
        
        st.markdown("---")
        st.write("# Data Requirement & Data Collection")

        st.write("""
        ### 
        Data yang digunakan dalam analisis ini diperoleh dari sumber resmi Badan Pusat Statistik (BPS) Yogyakarta melalui situs web [https://yogyakarta.bps.go.id/](https://yogyakarta.bps.go.id/).
        Informasi yang diambil mencakup data intensitas hujan dan hasil produksi padi di wilayah Daerah Istimewa Yogyakarta. 

        ### Variabel-Variabel Utama:
        - **Curah Hujan (mm³)**: Rata-rata curah hujan bulanan di setiap kabupaten/kota di DIY (Sleman, Bantul, Gunungkidul, Kulon Progo, dan Kota Yogyakarta).
        - **Bulan**: Waktu observasi data curah hujan dan hasil produksi padi. Data ini digunakan untuk mengidentifikasi pola musiman dan analisis dampak.
        - **Produksi Padi (ton)**: Volume padi yang dihasilkan per bulan di dalam satu tahun dalam satuan ton. Variabel ini merupakan target (dependent variable).

        ### Penggabungan Dataset:
        Proses analisis dilakukan dengan menggabungkan dua dataset yang berbeda:
        - **Data curah hujan** dan **data produksi padi**.
        Dataset digabungkan berdasarkan kolom waktu (bulan) agar analisis hubungan antara curah hujan dan produksi padi dapat dilakukan secara komprehensif.
        
        ### Ruang Lingkup Data:
        - **Wilayah**: DIY (Sleman, Bantul, Gunungkidul, Kulon Progo, dan Kota Yogyakarta).
        - **Periode**: Data dari tahun 2021–2023.
        """)

        st.markdown("---")
        st.write("# Data Preparation")

        st.write("""
        ### 
        Dalam analisis pengaruh intensitas hujan terhadap produksi padi di Daerah Istimewa Yogyakarta, fitur data yang akan digunakan untuk membangun 
        model Random Forest Regression adalah sebagai berikut:

        ### Variabel-Variabel Utama:
        - **Intensitas Hujan (CH)**: Jumlah curah hujan bulanan (dalam mm³). Variabel ini merupakan faktor utama yang mempengaruhi pertumbuhan tanaman padi..
        - **Bulan (Bln)**: Periode waktu observasi curah hujan dan hasil panen untuk analisis musiman.
        - **Produksi Padi (PP)**: Volume produksi padi dalam satuan ton per kabupaten/kota. Variabel ini adalah target (dependent variable) dalam analisis ini.
        """)

        st.markdown("---")
        st.write("# Modelling & Evaluation")

        st.write("""
        ### Model : Random Forest Regressor
        **Alasan**:
        - Dalam kasus prediksi jumlah produksi padi, variabel dependen adalah jumlah produksi padi, sedangkan variabel independen yang digunakan adalah intensitas hujan dan bulan.
        - Kami menggunakan model **Random Forest Regressor**, karena model ini mampu menangkap hubungan kompleks, termasuk hubungan non-linear antar variabel.
        - Random Forest juga memiliki **performa prediksi yang baik** dan **tahan terhadap overfitting**.

        ### Pembagian Data:
        - Total data yang digunakan adalah **96 baris** setelah data digabungkan dan diproses.
        - Data dibagi menjadi **80% untuk data latih** dan **20% untuk data uji**, menghasilkan **77 data latih** dan **19 data uji**.
        - Pembagian ini ideal untuk dataset moderat, memungkinkan generalisasi yang baik untuk model.

        ### Algoritma yang Dipilih:
        - Kami menggunakan algoritma **Random Forest** dengan parameter default:
            - `n_estimators=100` (menggunakan 100 decision trees)
            - `random_state=42` untuk hasil yang konsisten.
        - Setelah memisahkan data latih dan data uji, model dilatih dengan data latih dan digunakan untuk memprediksi data uji.

        ### Evaluasi Model:
        Evaluasi model dilakukan untuk mengukur kinerjanya dalam memprediksi produksi padi berdasarkan curah hujan dan waktu. Berikut metrik evaluasi yang digunakan:
        - **R-squared (R²):**
            - Memberikan informasi sejauh mana model dapat menjelaskan variasi dalam data target.
            - Nilai **R² berkisar antara 0 hingga 1**. Nilai lebih dekat ke 1 menunjukkan performa model yang lebih baik.
            - **R² membantu menggambarkan kemampuan variabel independen** dalam memprediksi variasi pada variabel dependen.
        - **Mean Squared Error (MSE):**
            - MSE menghitung rata-rata kuadrat kesalahan prediksi.
            - Nilai **MSE yang lebih kecil** menunjukkan prediksi yang lebih akurat.
            - **MSE membantu memahami tingkat akurasi prediksi** yang dihasilkan dan sejauh mana kesalahan prediksi dapat diminimalkan.
                 
        ### Hasil Evaluasi Sesuai Rencana Evaluasi :
        **Mean Squared Error**: 216.36866620874994
        **R^2 Score**: 0.7342465882223868
        Hasil evaluasi model Random Forest Regressor menunjukkan bahwa model memiliki kinerja yang baik dalam memprediksi produksi padi berdasarkan intensitas hujan dan waktu. 
        Berdasarkan matrik evaluasi, nilai Mean Squared Error (MSE) sebesar 216.37 menunjukkan rata-rata kesalahan prediksi yang masih dalam batas wajar, sementara nilai 
        R-squared (R²) sebesar 0.734 menunjukkan bahwa model mampu menjelaskan sekitar 73.4% variasi dalam data produksi padi. Hal ini menggambarkan bahwa variabel independen, 
        yaitu intensitas hujan dan bulan, memiliki kontribusi yang signifikan dalam memprediksi produksi padi, dengan model yang dapat diandalkan untuk analisis lebih lanjut.


        Dengan model ini, kami dapat menganalisis pengaruh curah hujan terhadap hasil produksi padi sekaligus membuat prediksi yang akurat untuk membantu perencanaan agrikultur.
        """)

        st.markdown("---")
        st.write("# Kesimpulan")

        st.write("""
        ### Model : Random Forest Regressor
        Hasil analisis menggunakan model Random Forest Regressor menunjukkan bahwa variabel intensitas hujan dan waktu memiliki pengaruh yang signifikan terhadap prediksi 
        produksi padi. Dengan nilai R-squared (R²) sebesar 0.734, model ini mampu menjelaskan sekitar 73.4% variasi dalam data produksi padi. Meskipun demikian, terdapat 
        ruang untuk peningkatan, khususnya dalam menangani variasi data yang belum sepenuhnya dapat dijelaskan oleh model. Salah satu kekurangan model ini adalah sensitivitas 
        terhadap jumlah data dan distribusinya, yang dapat mempengaruhi stabilitas prediksi terutama untuk dataset dengan outlier atau distribusi tidak merata. Namun, 
        kelebihan model ini terletak pada kemampuannya menangani hubungan non-linear antara variabel independen dan dependen dengan efisiensi komputasi yang baik.

        ### Strategi dan Rekomendasi:
        - Optimalisasi Waktu Tanam:
        Informasi prediksi produksi padi berdasarkan intensitas hujan dan bulan dapat digunakan petani untuk menentukan waktu tanam yang tepat sesuai pola cuaca. Dengan memanfaatkan data ini, risiko kerugian akibat anomali cuaca dapat diminimalkan.
        - Penyusunan Kebijakan Pangan Berbasis Data:
        Bagi pemerintah atau pengambil kebijakan, hasil analisis ini dapat digunakan untuk merancang kebijakan alokasi sumber daya seperti penyediaan air irigasi dan cadangan pangan, khususnya pada periode yang diprediksi akan mengalami penurunan produksi.
        - Pengelolaan Risiko Cuaca Ekstrem:
        Petani dapat diarahkan untuk memanfaatkan data prediksi dalam merencanakan diversifikasi tanaman, sehingga dampak dari cuaca ekstrem dapat diminimalkan. Misalnya, memilih varietas padi yang lebih tahan terhadap curah hujan tinggi atau kekeringan.
        - Penggunaan Teknologi Pertanian Berbasis Data:
        Disarankan untuk mengintegrasikan teknologi Internet of Things (IoT) dan data science untuk memperbarui informasi prediksi secara real-time. Hal ini dapat mempercepat respons terhadap perubahan cuaca dan pola intensitas hujan.

        ### Alasan Strategi dan Rekomendasi:
        Rekomendasi ini didasarkan pada pemahaman dari business understanding bahwa intensitas hujan dan pola musiman merupakan faktor utama dalam produksi padi. Dengan 
        menerapkan strategi berbasis data, petani dapat lebih efisien dalam mengelola hasil produksi, dan pemerintah dapat mendukung ketahanan pangan secara lebih efektif. 
        Optimalisasi hasil padi dan mitigasi risiko gagal panen juga diharapkan dapat meningkatkan kesejahteraan masyarakat agraris di wilayah Yogyakarta.
        """)
    else:
        st.warning("Harap unggah dataset terlebih dahulu di halaman 'Upload Data'.")


    