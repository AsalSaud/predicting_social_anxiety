import streamlit as st

# Page Config
st.set_page_config(page_icon=":email:", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', sans-serif;
    }
    h1 {
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px; /* Smaller margin to move Contact Us a bit up */
        margin-top: 10px;    /* Optional: add small top margin if needed */
    }
    /* Thin straight line */
    .thin-line {
        height: 1px; /* Very thin */
        width: 100%;
        background-color: #ffbd59; /* Exact color you asked */
        margin: 10px 0 40px 0; /* Space above and below */
        border-radius: 1px;
    }
    .member-card {
        text-align: center;
        padding: 10px;
    }
    .member-card img {
        width: 130px;
        height: 130px;
        object-fit: cover;
        border-radius: 50%;
        background-color: white;
        margin-bottom: 15px;
    }
    .member-name {
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 5px;
    }
    .member-email {
        font-size: 14px;
        color: #555;
        line-height: 1.2;
        margin-bottom: 5px;
    }
    .linkedin-link {
        color: #0077b5;
        text-decoration: none;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Contact Us</h1>", unsafe_allow_html=True)

st.markdown('<div class="thin-line"></div>', unsafe_allow_html=True)


team = [
    {
        "name": "Asal Almughamisi",
        "email": "Asaalsaud06@gmail.com",
        "linkedin": "https://www.linkedin.com/in/asal-almughamisi-90710124a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app",
        "photo": "https://i.imgur.com/5RXUA2l.png"
    },
    {
        "name": "Asma Alfaify",
        "email": "asmaalfaify5@gmail.com",
        "linkedin": "https://www.linkedin.com/in/asmaa-alfaify-288867342?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app",
        "photo": "https://i.imgur.com/5RXUA2l.png"
    },
    {
        "name": "Sondus Alenazi",
        "email": "Sondus Alenazi@gmail.com",
        "linkedin": "https://www.linkedin.com/in/sondus-alenazi-92935a234/",
        "photo": "https://i.imgur.com/5RXUA2l.png"
    },
    {
        "name": "Danah Salem",
        "email": "danatalkhalij22@gmail.com",
        "linkedin": "https://www.linkedin.com/in/danahsalem/",
        "photo": "https://i.imgur.com/5RXUA2l.png"
    },
    {
        "name": "Seham Matouqe",
        "email": "sehammatouqe@outlook.sa",
        "linkedin": "https://www.linkedin.com/in/seham-matouqe-46bb79280?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app",
        "photo": "https://i.imgur.com/5RXUA2l.png"
    },
]


cols = st.columns(5)


for idx, member in enumerate(team):
    with cols[idx]:
        st.markdown(f"""
            <div class="member-card">
                <img src="{member['photo']}">
                <div class="member-name">{member['name']}</div>
                <div class="member-email">{member['email'].split('@')[0]}<br>{member['email'].split('@')[1]}</div>
                <a href="{member['linkedin']}" class="linkedin-link" target="_blank">LinkedIn Profile</a>
            </div>
        """, unsafe_allow_html=True)
