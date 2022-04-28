#include <iostream>
#include <chrono>
#include <Eigen/Dense>

#include "TLK.h"
#include "Engine.h"

#include <Windows.h>

#include <gl/GL.h>
#include <gl/GLU.h>

#include <dwmapi.h>

struct Drone
{
    size_t agent;
    float left;
    float right;
    Engine::Vec2 leftDir;
    Engine::Vec2 rightDir;

    float fitness;
};

TLK::Model m{};
Engine::World world{};
int w = 1920;
int h = 1080;

float goalTimer;

float timer;

float compute = 0;
const float freq = 10;

const size_t population = 2000;
std::vector<Drone> drones;

Engine::Vec2 goal;

void Init()
{
    std::srand((unsigned int)time(NULL));

    m.Append(TLK::Layer(TLK::LSTM, TLK::Tensor(8, 1), TLK::Tensor(10, 1)));
    m.Append(TLK::Layer(TLK::LSTM, TLK::Tensor(10, 1), TLK::Tensor(4, 1)));
    m.Compile();
    m.Agent(population);

    m.Load("test2.arby");

    drones.reserve(population);

    goal = { w / 2.0f, h / 2.0f };

    for (size_t i = 0; i < population; ++i)
    {
        drones.push_back({});
        Drone& d = drones[i];

        d.agent = i;
        d.left = 0;
        d.right = 0;

        //Engine::Vec2 pos{ std::rand() / float(RAND_MAX) * w, std::rand() / float(RAND_MAX) * h };
        //Engine::Vec2 pos = Engine::zero;
        Engine::Vec2 pos = { w / 2.0f, h / 2.0f };

        world.dynamics.Add(Engine::Circle(2.0f, { pos.x - 14.0f, pos.y }, 200));
        world.dynamics.Add(Engine::Circle(4.0f, pos, 0));
        world.dynamics.Add(Engine::Circle(2.0f, { pos.x + 14.0f, pos.y }, 200));

        size_t worldIndex = i * 3;

        world.joints.push_back(Engine::Joint{ &world.dynamics.items[worldIndex].value, &world.dynamics.items[worldIndex + 1].value, 14.0f, 0.3f, 0.0f, 0.5f });
        world.joints.push_back(Engine::Joint{ &world.dynamics.items[worldIndex + 2].value, &world.dynamics.items[worldIndex + 1].value, 14.0f, 0.3f, 0.0f, 0.5f });
        world.joints.push_back(Engine::Joint{ &world.dynamics.items[worldIndex].value, &world.dynamics.items[worldIndex + 2].value, 28.0f, 0.5f, 0.0f, 0.5f });
    }

    world.Step(0.01f);
    world.Step(0.01f);
    world.Step(0.01f);
}

inline Engine::Vec2 NormalizeDir(Engine::Vec2& a, Engine::Vec2& b)
{
    Engine::Vec2 direction{ b.x - a.x, b.y - a.y };
    float distance = std::sqrtf(direction.x * direction.x + direction.y * direction.y);
    if (distance != 0)
        return { direction.x / distance, direction.y / distance };
    else
        return { 0, 1 };
}

inline Engine::Vec2 Dir(Engine::Vec2& a, Engine::Vec2& b)
{
    return { b.x - a.x, b.y - a.y };
}

inline Engine::Vec2 Normalize(Engine::Vec2 direction)
{
    float distance = std::sqrtf(direction.x * direction.x + direction.y * direction.y);
    if (distance != 0)
        return { direction.x / distance, direction.y / distance };
    else
        return { 0, 1 };
}

void Update(float dt)
{
    Engine::AllocatorContainer<Engine::Circle>* dynamics = world.dynamics.items;

    float rotate = 3 * dt;
    float power = 450 * dt;

    timer += dt;
    compute += dt;

    if (compute >= 1.0f / freq)
    {
        compute = 0;
        for (size_t i = 0; i < population; ++i)
        {
            Drone& d = drones[i];
            size_t worldIndex = d.agent * 3;

            Eigen::MatrixXf& inputs = m.inputs[d.agent];

            Engine::Vec2 dir2goal = Dir(dynamics[worldIndex + 1].value.position, goal);
            Engine::Vec2 norm = Normalize(dir2goal);
            Engine::Vec2 body = NormalizeDir(dynamics[worldIndex + 1].value.position, dynamics[worldIndex + 2].value.position);

            float dot = Engine::right.x * body.x + Engine::right.y * body.y;
            float det = Engine::right.x * body.y - Engine::right.y * body.x;
            float angle = std::atan2(det, dot);

            d.fitness += std::abs(angle);

            inputs <<
                norm.x,
                norm.y,
                std::cosf(angle),
                std::sinf(angle),
                std::cosf(d.left),
                std::sinf(d.left),
                std::cosf(d.right),
                std::sinf(d.right);
        }
        m.Compute();
    }

    for (size_t i = 0; i < population; ++i)
    {
        Drone& d = drones[i];
        size_t worldIndex = d.agent * 3;

        Eigen::Map<Eigen::MatrixXf>& outputs = m.outputs[d.agent];

        //d.left = 3.141592654f * ((outputs(0) + 1.0f) / 2.0f);
        //d.right = 3.141592654f * ((outputs(1) + 1.0f) / 2.0f);

        d.left += 3 * outputs(0) * dt;
        d.right += 3 * outputs(1) * dt;

        Engine::Vec2 left = Dir(dynamics[worldIndex + 1].value.position, dynamics[worldIndex].value.position);
        Engine::Vec2 right = Dir(dynamics[worldIndex + 2].value.position, dynamics[worldIndex].value.position);

        d.leftDir = Normalize({
            left.y * std::cosf(d.left) + left.x * std::sinf(d.left),
            left.y * std::sinf(d.left) - left.x * std::cosf(d.left)
            });

        d.rightDir = Normalize({
            right.y * std::cosf(d.right) + right.x * std::sinf(d.right),
            right.y * std::sinf(d.right) - right.x * std::cosf(d.right)
            });

        float lpower = power * ((outputs(2) + 1.0f) / 2.0f);
        float rpower = power * ((outputs(3) + 1.0f) / 2.0f);

        dynamics[worldIndex].value.velocity.x += d.leftDir.x * lpower;
        dynamics[worldIndex].value.velocity.y += d.leftDir.y * lpower;

        dynamics[worldIndex + 2].value.velocity.x += d.rightDir.x * rpower;
        dynamics[worldIndex + 2].value.velocity.y += d.rightDir.y * rpower;
    }

    goalTimer += dt;
    if (goalTimer > 20.0f)
    {
        goalTimer = 0;
        goal = { std::rand() / float(RAND_MAX) * w, std::rand() / float(RAND_MAX) * h };

        for (size_t i = 0; i < population; ++i)
        {
            Drone& d = drones[i];
            size_t worldIndex = d.agent * 3;

            Engine::Vec2 dir2goal = Dir(dynamics[worldIndex + 1].value.position, goal);
            d.fitness += std::sqrtf(dir2goal.x * dir2goal.x + dir2goal.y * dir2goal.y);
        }
    }

    if (timer > 60.0f)
    {
        m.Save("test2.arby");

        timer = 0;
        std::sort(drones.begin(), drones.end(), [](const Drone& a, const Drone& b)
            {
                return (a.fitness > b.fitness);
            });

        std::cout << "best: " << drones.back().fitness << " worst: " << drones[0].fitness << std::endl;

        for (size_t i = 0; i < population; ++i)
        {
            Drone& d = drones[i];
            size_t worldIndex = d.agent * 3;

            //Engine::Vec2 pos{ std::rand() / float(RAND_MAX) * w, std::rand() / float(RAND_MAX) * h };
            //Engine::Vec2 pos = { w / 2, h / 2 };
            //Engine::Vec2 pos = dynamics[o.agent * 3 + 1].value.position;

            /*dynamics[worldIndex].value.position = {pos.x - 14.0f, pos.y};
            dynamics[worldIndex].value.velocity = Engine::zero;//dynamics[o.agent * 3].value.velocity;
            dynamics[worldIndex + 1].value.position = pos;
            dynamics[worldIndex + 1].value.velocity = Engine::zero; //dynamics[o.agent * 3 + 1].value.velocity;
            dynamics[worldIndex + 2].value.position = { pos.x + 14.0f, pos.y };
            dynamics[worldIndex + 2].value.velocity = Engine::zero; //dynamics[o.agent * 3 + 2].value.velocity;*/

            d.fitness = 0;

            d.left = 0;// o.left;
            d.right = 0;// o.right;

            Engine::Vec2 pos = { w / 2, h / 2 }; //dynamics[o.agent * 3 + 1].value.position;

            dynamics[worldIndex].value.position = { pos.x - 14.0f, pos.y };
            dynamics[worldIndex].value.velocity = Engine::zero;
            dynamics[worldIndex + 1].value.position = pos;
            dynamics[worldIndex + 1].value.velocity = Engine::zero;
            dynamics[worldIndex + 2].value.position = { pos.x + 14.0f, pos.y };
            dynamics[worldIndex + 2].value.velocity = Engine::zero;

            size_t offset = 3 * population / 4;
            if (i < 3 * population / 4)
            {
                size_t index = offset - 1 + (int)(std::rand() / float(RAND_MAX) * (population / 4));
                Drone& o = drones[index];

                m.Copy(d.agent, index);
                m.Mutate(d.agent);
                m.Reset(i);
            }
        }
    }

    world.Step(dt);
}

void DrawCircle(float ori_x, float ori_y, float radius)
{
    glBegin(GL_POLYGON);
    int resolution = 20;
    for (int i = 0; i <= resolution; i++) {
        float angle = 2.0f * 3.141592654f * i / resolution;
        float x = cos(angle) * radius;
        float y = sin(angle) * radius;
        glVertex2d(ori_x + x, ori_y + y);
    }
    glEnd();
}

HGLRC m_hrc;

BOOL initSC() {
    glEnable(GL_ALPHA_TEST);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0, 0, 0, 0);

    return 0;
}

void resizeSC(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

BOOL Render() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

    /*glBegin(GL_TRIANGLES);                              // Drawing Using Triangles
    glColor4f(1.0f, 0.0f, 0.0f, 0.5f);                      // Set The Color To Red
    glVertex3f(0.0f, 1.0f, 0.0f);                  // Top
    glColor3f(0.0f, 1.0f, 0.0f);                      // Set The Color To Green
    glVertex3f(-1.0f, -1.0f, 0.0f);                  // Bottom Left
    glColor3f(0.0f, 0.0f, 1.0f);                      // Set The Color To Blue
    glVertex3f(1.0f, -1.0f, 0.0f);                  // Bottom Right
    glEnd();*/

    glLoadIdentity();                           // Reset The Projection Matrix
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glOrtho(0, w, 0, h, -1, 1);                      // Set Up An Ortho Screen

    glTranslated(0, 0, 0);
    glScaled(1, 1, 0); // Scales from bottom left *bruh*

    glColor4f(1.0f, 0.2f, 1.0f, 0.6f);
    DrawCircle(goal.x, goal.y, 6);

    for (size_t i = 0; i < world.dynamics.last(); ++i)
    {
        if (world.dynamics.items[i].active)
        {
            glColor4f(0.0f, 0.8f, 0.2f, 0.6f);
            Engine::Circle& c = world.dynamics.items[i].value;
            DrawCircle(c.position.x, c.position.y, c.radius);
        }
    }

    Engine::AllocatorContainer<Engine::Circle>* dynamics = world.dynamics.items;
    for (size_t i = 0; i < population; ++i)
    {
        Drone& d = drones[i];
        size_t worldIndex = d.agent * 3;
        Engine::Circle l = dynamics[worldIndex].value;
        Engine::Circle r = dynamics[worldIndex + 2].value;

        glColor4f(0.8f, 0.0f, 0.2f, 0.6f);
        
        DrawCircle(l.position.x + d.leftDir.x * 7.0f, l.position.y + d.leftDir.y * 7.0f, 2.0f);
        DrawCircle(l.position.x - d.leftDir.x * 7.0f, l.position.y - d.leftDir.y * 7.0f, 2.0f);

        DrawCircle(r.position.x + d.rightDir.x * 7.0f, r.position.y + d.rightDir.y * 7.0f, 2.0f);
        DrawCircle(r.position.x - d.rightDir.x * 7.0f, r.position.y - d.rightDir.y * 7.0f, 2.0f);
    }

    glPopMatrix();
    glFlush();

    return 0;
}

BOOL CreateHGLRC(HWND hWnd) {
    PIXELFORMATDESCRIPTOR pfd = {
      sizeof(PIXELFORMATDESCRIPTOR),
      1,                                // Version Number
      PFD_DRAW_TO_WINDOW |         // Format Must Support Window
      PFD_SUPPORT_OPENGL |         // Format Must Support OpenGL
      PFD_SUPPORT_COMPOSITION |         // Format Must Support Composition
      PFD_DOUBLEBUFFER,                 // Must Support Double Buffering
      PFD_TYPE_RGBA,                    // Request An RGBA Format
      32,                               // Select Our Color Depth
      0, 0, 0, 0, 0, 0,                 // Color Bits Ignored
      8,                                // An Alpha Buffer
      0,                                // Shift Bit Ignored
      0,                                // No Accumulation Buffer
      0, 0, 0, 0,                       // Accumulation Bits Ignored
      24,                               // 16Bit Z-Buffer (Depth Buffer)
      8,                                // Some Stencil Buffer
      0,                                // No Auxiliary Buffer
      PFD_MAIN_PLANE,                   // Main Drawing Layer
      0,                                // Reserved
      0, 0, 0                           // Layer Masks Ignored
    };

    HDC hdc = GetDC(hWnd);
    int PixelFormat = ChoosePixelFormat(hdc, &pfd);
    if (PixelFormat == 0) {
        return FALSE;
    }

    BOOL bResult = SetPixelFormat(hdc, PixelFormat, &pfd);
    if (bResult == FALSE) {
        return FALSE;
    }

    m_hrc = wglCreateContext(hdc);
    if (!m_hrc) {
        return FALSE;
    }

    ReleaseDC(hWnd, hdc);

    return TRUE;
}


int running = 1;

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
    switch (message)
    {
    case WM_CHAR:
        if (wparam == VK_ESCAPE)
        {
            running = 0;
            DestroyWindow(hwnd);
        }
        return 0;
    case WM_DESTROY:
        if (m_hrc)
        {
            wglMakeCurrent(NULL, NULL);
            wglDeleteContext(m_hrc);
        }
        running = 0;
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProc(hwnd, message, wparam, lparam);
    }
}

int main()
{
    //HWND hWnd = GetConsoleWindow();
    //ShowWindow(hWnd, SW_MINIMIZE);
    //ShowWindow(hWnd, SW_HIDE);

    const char* windowClassName = "Window in Console";
    WNDCLASS windowClass = { 0 };
    windowClass.hbrBackground = (HBRUSH)CreateSolidBrush(0x00000000);
    windowClass.hCursor = LoadCursor(NULL, IDC_ARROW);
    windowClass.hInstance = NULL;
    windowClass.lpfnWndProc = WndProc;
    windowClass.lpszClassName = windowClassName;
    windowClass.style = CS_HREDRAW | CS_VREDRAW;
    if (!RegisterClass(&windowClass))
        MessageBox(NULL, "Could not register class", "Error", MB_OK);

    w = GetSystemMetrics(SM_CXSCREEN);
    h = GetSystemMetrics(SM_CYSCREEN);

    HWND windowHandle = CreateWindowEx(
        WS_EX_COMPOSITED | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
        windowClassName,
        NULL,
        WS_POPUP, //borderless
        0, //x coordinate of window start point
        0, //y start point
        w, //width of window; this function
        h, //height of the window
        NULL, //handles and such, not needed
        NULL,
        NULL,
        NULL);
    ShowWindow(windowHandle, SW_MAXIMIZE);
    SetLayeredWindowAttributes(windowHandle, 0, 255, LWA_ALPHA);

    DWM_BLURBEHIND bb = { 0 };
    HRGN hRgn = CreateRectRgn(0, 0, -1, -1);
    bb.dwFlags = DWM_BB_ENABLE | DWM_BB_BLURREGION;
    bb.hRgnBlur = hRgn;
    bb.fEnable = TRUE;
    DwmEnableBlurBehindWindow(windowHandle, &bb);

    SetLayeredWindowAttributes(windowHandle, 0, 255, LWA_ALPHA);

    CreateHGLRC(windowHandle);

    HDC hdc = GetDC(windowHandle);
    wglMakeCurrent(hdc, m_hrc);
    initSC();
    resizeSC(w, h);

    MSG messages;

    Init();

    auto prev = std::chrono::high_resolution_clock::now();

    while (running)
    {
        if (PeekMessage(&messages, NULL, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&messages);
            DispatchMessage(&messages);
        }
        else
        {
            /*auto now = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float, std::milli>(now - prev).count() / 1000.0f;
            prev = now;*/

            //std::cout << dt << std::endl;
            Update(0.01);
            //Update(1.0f / freq);
            Render();
            SwapBuffers(hdc);
        }
    }

    ReleaseDC(windowHandle, hdc);
    DeleteObject(windowHandle); //doing it just in case

    return messages.wParam;
}