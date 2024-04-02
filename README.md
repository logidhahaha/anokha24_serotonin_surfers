# Team Name: anokha24_serotonin_surfers
🌊 Welcome to the TEAM Serotonin Surfers! 🏄‍♂️
We are a dynamic group of innovators who ride the waves of creativity to craft cutting-edge solutions. With a passion for technology and a drive for excellence, we've proudly clinched few prestigious hackathon prizes. Our team is dedicated to smart working, leveraging our diverse skills and expertise to deliver impactful results. 

🌟 **PROBLEM STATEMENT**: 

The challenge is to develop a real-time sign language translation system that accurately converts Indian Sign Language (ISL) gestures into spoken language or text. Sign language serves as a primary communication mode for many individuals with hearing impairments, yet it can pose a barrier to communication with those who do not understand it. Hence, creating a system capable of automatically translating sign language into spoken language or text can greatly enhance communication accessibility for the deaf and hard of hearing community.

💡 **SOLUTION**: 

![WhatsApp Image 2024-04-02 at 7 38 50 PM](https://github.com/logidhahaha/anokha24_serotonin_surfers/assets/121540092/d3359fd3-ae42-4836-99f9-d0643be2b1d3)

Our proposed solution encompasses several key components aimed at developing an efficient and accurate system for Indian Sign Language (ISL) recognition. By acquiring diverse ISL image data and extracting relevant features such as hand location, orientation, and finger movements, we lay the groundwork for training a deep learning model capable of recognizing ISL gestures. This model is then optimized using <b>OpenVINO</b>, ensuring efficient execution on <b>Intel hardware</b> platforms. The development of an application using OpenVINO enables real-time video capture, feature processing, and translation of ISL signs into text or speech, thereby facilitating communication accessibility for individuals with hearing impairments. Continuous evaluation and refinement of the system are essential for improving accuracy and real-time performance, ensuring that it meets the evolving needs of its users.

🎯 **Normal model Vs Intel optimized**:

![performance](https://github.com/logidhahaha/anokha24_serotonin_surfers/assets/121540092/a0dabf7a-d10e-4b78-9f1d-ba354f03d12c)



🚀 **Project Demo**:

<div align="center">
  <a href="https://www.youtube.com/watch?v=_JYYyQgaWrs"><img src="https://img.youtube.com/vi/_JYYyQgaWrs/0.jpg" alt="IMAGE ALT TEXT"></a>
</div>



🎯 **INTEL ONE API OPENVINO TOOLKIT**:

We have used the Intel oneAPI OpenVINO toolkit to enhance the efficiency and performance of our sign language translation system. Initially, we employed the toolkit's Model Optimizer tool to convert and optimize our trained deep learning model for inference, ensuring compatibility with Intel hardware platforms. By harnessing the power of Intel's hardware acceleration technologies, including Intel Deep Learning Boost and Intel Distribution of OpenVINO, we achieved real-time performance and minimized computational resources while executing the sign language translation system. 

⚙️ **USECASE OF INTEL DEVELOPER CLOUD**:

In our project, we optimized the sign language translation system by leveraging the Intel Developer Cloud's advanced computing infrastructure and optimization tools. We utilized the cloud's diverse range of virtual machines with varying CPU and GPU configurations to test our code across different environments, ensuring compatibility and performance consistency. Through iterative testing and validation, we fine-tuned our code to maximize efficiency and resource utilization, particularly by deploying it on virtual instances equipped with Intel's hardware acceleration technologies. By harnessing the power of the Intel Developer Cloud, we were able to optimize our sign language translation system for optimal performance on Intel hardware platforms, ensuring its readiness for deployment in real-world scenarios.

🚀 **FUTURE SCOPE**: 

The future scope for the sign language translation project is broad and promising, with opportunities for advancement in various aspects. Enhancements in gesture recognition algorithms will lead to better understanding and interpretation of complex sign language gestures, including nuances in hand movements and facial expressions. Further development could expand the system to support multiple sign languages, promoting inclusivity on a global scale. Real-time collaboration features would facilitate remote communication between sign language users and non-signers, while mobile and wearable applications could provide on-the-go access to translation services. Integrating with existing accessibility tools and platforms would enhance accessibility, while educational and training tools could support sign language learning initiatives. Continuous engagement with the deaf and hard of hearing community will be crucial for gathering feedback and ensuring that the system evolves to meet the diverse needs of its users, ultimately fostering communication accessibility and empowerment.

🧠 **LEARNINGS & INSIGHTS:**

- **Technological Empowerment:** It underscores the transformative power of technology in empowering individuals with disabilities. By developing a sign language translation system, the project demonstrates how advanced technologies like deep learning and computer vision can break communication barriers and enhance accessibility for people with hearing impairments.
- **Cross-Disciplinary Collaboration:** The project highlights the importance of collaboration across disciplines such as computer science, linguistics, and assistive technology. Bringing together expertise from diverse fields is essential for understanding the complexities of sign language, designing effective translation algorithms, and ensuring cultural sensitivity in the development process.
- **User-Centric Design:** Through continuous engagement with end-users and stakeholders, the project emphasizes the importance of user-centric design. Gathering feedback and insights from the deaf and hard of hearing community helps identify usability issues, refine translation accuracy, and enhance the overall user experience of the system.
- **Ethical Considerations:** Developing assistive technologies like sign language translation systems requires careful consideration of ethical issues such as privacy, data security, and cultural sensitivity. Addressing these concerns is critical for building trust with users and ensuring the responsible deployment of the technology.
- **Impact on Accessibility:** Ultimately, the project highlights the significant impact that accessible technology can have on improving the lives of individuals with disabilities. By enabling real-time translation of sign language into spoken language or text, the system promotes inclusivity, independence, and equal participation in society for people with hearing impairments.

💻 **TECH STACKS:**

  ![techstack(2)](https://github.com/logidhahaha/anokha24_serotonin_surfers/assets/121540092/b1a172ac-2178-46a7-843a-562a613900e5)
  

🌈 **CONCLUSION**:

The sign language translation project represents a significant step forward in leveraging technology to enhance accessibility and inclusivity for individuals with hearing impairments. By combining advanced computer vision techniques, deep learning algorithms, and user-centric design principles, the project has developed a system capable of real-time translation of Indian Sign Language (ISL) gestures into spoken language or text. By breaking down communication barriers and fostering understanding between sign language users and non-signers, the project exemplifies the transformative potential of accessible technology in creating a more inclusive and equitable society.


🔗 CLONE REPOSITORY:

```
git clone https://github.com/logidhahaha/anokha24_serotonin_surfers.git
```

🛠️ **LIBRARIES REQUIRED**:
```
pip install Flask
pip install opencv-python
pip install numpy
pip install tensorflow
pip install notebook
pip install openvino
```
[Install OpenVino Toolkit](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_overview.html)

💻 **Application:**

```
-anokha24_serotonin_surfers
  - model.py
  - main.py
  - frozen_graph.xml
  - frozen_graph.bin
-anokha24_serotonin_surfers/ISL_Dataset
-anokha24_serotonin_surfers/templates
  -index.html
```
💻 **To run the application in IDC(Intel Developer Cloud):**

```
>ssh -<user> guest@<ip> ubuntu@<ip>

>mkdir anokha24_serotonin_surfers
>cd anokha24_serotonin_surfers

>git clone https://github.com/logidhahaha/anokha24_serotonin_surfers.git

>python main.py

>snap install ngrok
>ngrok http <port number>
```













