#pragma once

template <typename Ty>
class Singleton {
public:
    static Ty* get() {
        if (!instance) {
            instance = new (std::nothrow) Ty();
        }
        return instance;
    }

    static void release() {
        if (instance)
            delete instance;
        instance = nullptr;
    }

    static bool isAlive() {
        return (instance != nullptr);
    }

    

private:
    static Ty* instance; 
};

template <typename Ty>
Ty* Singleton<Ty>::instance = nullptr;

