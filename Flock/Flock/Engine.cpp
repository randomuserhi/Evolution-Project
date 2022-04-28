#include "Engine.h"
#include <cassert>

namespace Engine
{
	Vec2 zero{ 0, 0 };
	Vec2 right{ 1, 0 };

	void World::Step(float dt)
	{
		size_t snapIdx = 0;
		for (size_t i = 0; i < dynamics.last(); ++i)
		{
			if (dynamics.items[i].active)
			{
				Circle& c = dynamics.items[i].value;

				c.velocity.y -= c.gravity * dt;

				c.position.x += c.velocity.x * dt;
				c.position.y += c.velocity.y * dt;

				float border = 0;

				/*if (c.position.y < 0 - border)
				{
					c.position.y = 0 - border;
					if (c.velocity.y < 0) c.velocity.y = 0;
				}
				else if (c.position.y > 1440 + border)
				{
					c.position.y = 1440 + border;
					if (c.velocity.y > 0) c.velocity.y = 0;
				}

				if (c.position.x < 0 - border)
				{
					c.position.x = 0 - border;
					if (c.velocity.x < 0) c.velocity.x = 0;
				}
				else if (c.position.x > 2560 + border)
				{
					c.position.x = 2560 + border;
					if (c.velocity.x > 0) c.velocity.x = 0;
				}*/
			}
		}
		for (size_t i = 0; i < joints.size(); ++i)
		{
			Joint& j = joints[i];
			Vec2 direction{ j.b->position.x - j.a->position.x, j.b->position.y - j.a->position.y };
			float displacement = std::sqrtf(direction.x * direction.x + direction.y * direction.y);
			float x = displacement - j.distance;

			Vec2 normal{ 0, 1 };
			if (displacement != 0) normal = { direction.x / displacement, direction.y / displacement };

			Vec2 relativeVel{ j.b->velocity.x - j.a->velocity.x, j.b->velocity.y - j.a->velocity.y };
			float relativeVelNorm = normal.x * relativeVel.x + normal.y * relativeVel.y;

			const float invMass = 1;
			float impulse = -(relativeVelNorm + j.beta * x / dt) * (1 / (j.gamma + invMass));

			assert(j.ratio >= 0 && j.ratio <= 1);

			j.a->velocity.x -= normal.x * impulse * j.ratio;
			j.a->velocity.y -= normal.y * impulse * j.ratio;
			j.b->velocity.x += normal.x * impulse * (1 - j.ratio);
			j.b->velocity.y += normal.y * impulse * (1 - j.ratio);
		}
	}
}