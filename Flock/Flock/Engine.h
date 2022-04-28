#pragma once

#include <iostream>
#include <vector>
#include <stack>

namespace Engine
{
	struct Vec2
	{
		float x;
		float y;
	};

	extern Vec2 zero;
	extern Vec2 right;

	struct Circle
	{
		float gravity;
		float radius;
		Vec2 position;
		Vec2 velocity;

		Circle()
		{
			radius = 0.5f;
			position = zero;
			velocity = zero;
		}
		Circle(float rad, Vec2 pos, float grav)
		{
			gravity = grav;
			radius = rad;
			position = pos;
			velocity = zero;
		}
	};

	struct Joint
	{
		Circle* a;
		Circle* b;

		float distance;
		float beta;
		float gamma;
		float ratio;
	};

	template<typename T>
	struct AllocatorContainer
	{
		bool active = false;
		T value;
	};

	template<typename T>
	struct Allocation
	{
		AllocatorContainer<T>* items;
		std::stack<size_t> available;

		Allocation(size_t count)
		{
			items = new AllocatorContainer<T>[count];
			for (size_t i = count; i > 0; --i)
			{
				available.push(i - 1);
			}
		}
		~Allocation()
		{
			delete[] items;
		}

		inline size_t last()
		{
			return _last + 1;
		}

		inline void Add(T item)
		{
			size_t index = available.top();
			available.pop();
			if (index > _last) _last = index;
			AllocatorContainer<T>& container = items[index];
			assert(!container.active);
			container.active = true;
			container.value = item;
		}
		inline void RemoveAt(size_t i)
		{
			AllocatorContainer<T>& container = items[i];
			assert(container.active);
			container.active = false;
			available.push(i);
			if (i == _last) while (!items[--_last].active && _last != 0) {}
		}

	private:
		size_t _last = 0;
	};

	struct World
	{
		Allocation<Circle> statics;
		Allocation<Circle> dynamics;
		std::vector<Joint> joints;

		World() : statics(10000), dynamics(10000)
		{
		}

		void Step(float dt);
	};
}