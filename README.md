# Project_ATS - Attendance Tracking System
ATS is an attendance tracking system developed using Python and OpenCV. It makes use if facial recognition technology to ensure accurate and transparent attendance records.

## Installation and Setup

### 1. Clone the Repository

```sh
git clone https://github.com/psaishya/Project_ATS.git
cd Project_ATS
```

### 2. Create and Activate Virtual Environment (Optional but Recommended)

#### On Windows:

```sh
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```sh
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Change Directory

```sh
cd ats
```

### 5. Apply Migrations

```sh
python manage.py migrate
```


### 6. Run the Development Server

```sh
python manage.py runserver
```

By default, the project will run at:

```
http://127.0.0.1:8000/
```


