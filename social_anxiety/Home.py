import streamlit as st
from PIL import Image
st.markdown(
    """
    <h1 style='text-align: center; color: #4B3832; display: inline;'>
        Welcome to Anxiety Explorer!
    </h1>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns([9, 1])
with col2:
    logo = ("https://i.imgur.com/K1rwgKG.png")
    st.image(logo, width=150)
st.markdown(
    "<h3 style='text-align: left; color: #6B4226;'>Step into the world of emotions ☁️ </h3>",
    unsafe_allow_html=True
)
st.markdown("---")
st.markdown(
    "<h2 style='color: #704214;'> Unsean Struggles: Mapping the Invisible Walls of Social Anxiety 😟🚪 </h2>",
    unsafe_allow_html=True
)
st.write(
    "Imagine you’re in a room full of people… your heart is racing, "
    "your hands are sweating, and your mind is screaming with nonstop thoughts: "
    "'What if I make a mistake? What if they laugh at me?'\n\n"
    "This isn’t just a simple fear of speaking in front of others. "
    "This is social anxiety — a silent enemy that robs millions of the simplest moments of daily life: "
    "from meeting a new friend to applying for the job of your dreams.\n\n"
    "And just like the character Anxiety in Inside Out 2, "
    "it represents the inner turmoil we experience, those feelings we can’t see but live with every day. "
    "Anxiety is that force that makes us feel not good enough, "
    "as if we could be judged for failure or embarrassment."
)
st.markdown("---")
st.markdown(
    "<h2 style='color: #704214;'> The Challenge 🚀🔥 </h2>",
    unsafe_allow_html=True
)
st.write(
    "Social Anxiety often hides behind a smile. "
    "It’s not always obvious — and sadly, many struggle silently without seeking help. "
    "By the time they reach out, it might already be overwhelming."
)
st.markdown("---")
st.markdown(
    "<h2 style='color: #704214;'> Conquering Social Anxiety Together 💡⭐️🤩 </h2>",
    unsafe_allow_html=True
)
st.write(
    "That’s why we built this project! By using ML and real-world data "
    "(like age, sleep patterns, caffeine intake, and stress levels and etc), "
    "we predict social anxiety levels early — empowering people to seek help before it's too late."
)
st.markdown("---")
if st.button(":rocket: Let's play together and Discover Your Score Index"):
    st.switch_page("app")
