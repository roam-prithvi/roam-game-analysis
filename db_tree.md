# Tree in DB

Successfully generated tree plan.
Selector, Sequence, BTConditionalInRange, BTForArrayElement, BTActionMoveAway, Sequence, BTConditionalInRange, BTForArrayElement, BTActionMoveTowards, Parallel, BTActionRotate, BTActionMoveTowards
_______________________
NodeLocation: 1
NodeType: Selector
NodeFunction: Root node that selects one of three behaviors based on the proximity of other objects. It prioritizes moving away from very close objects, then moving towards objects in an optimal range, and finally defaults to an idle behavior.
NodeRequirements:
NodeChildren:
	NodeLocation: 1.1
	NodeType: Sequence
	NodeFunction: This sequence handles the 'too close' scenario. It first checks for objects within a 2-unit radius and then moves away from the closest one found.
	NodeRequirements:
	NodeChildren:
		NodeLocation: 1.1.1
		NodeType: BTConditionalInRange
		NodeFunction: Checks for any object within a range of 0 to 2 units. If an object is found, it succeeds and provides a list of these objects, sorted by distance.
		NodeRequirements:
		NodeChildren:
		NodeLocation: 1.1.2
		NodeType: BTForArrayElement
		NodeFunction: Decorator that extracts the first element (index 0) from the list of nearby objects provided by node 1.1.1. This isolates the closest object for the move action.
		NodeRequirements: Requires the 'allObjectsInRange' list from node 1.1.1.
		NodeChildren:
			NodeLocation: 1.1.2.1
			NodeType: BTActionMoveAway
			NodeFunction: Moves the agent away from the target object identified by node 1.1.2. The movement is reactive, not run to completion, so the 'runTillTargetIsReached' field must be set to 'false'.
			NodeRequirements: Requires the 'element' transform provided by the parent decorator node 1.1.2.
			NodeChildren:
	NodeLocation: 1.2
	NodeType: Sequence
	NodeFunction: This sequence handles the 'follow' scenario. It checks for objects in the 5 to 8 unit range and then moves towards the closest one.
	NodeRequirements:
	NodeChildren:
		NodeLocation: 1.2.1
		NodeType: BTConditionalInRange
		NodeFunction: Checks for any object within a range of 5 to 8 units. If an object is found, it succeeds and provides a list of these objects.
		NodeRequirements:
		NodeChildren:
		NodeLocation: 1.2.2
		NodeType: BTForArrayElement
		NodeFunction: Decorator that extracts the first element (index 0) from the list of objects provided by node 1.2.1, identifying the closest target to follow.
		NodeRequirements: Requires the 'allObjectsInRange' list from node 1.2.1.
		NodeChildren:
			NodeLocation: 1.2.2.1
			NodeType: BTActionMoveTowards
			NodeFunction: Moves the agent towards the target object identified by node 1.2.2 at a speed of 3 units/second. The movement is reactive, so the 'runTillTargetIsReached' field must be set to 'false'.
			NodeRequirements: Requires the 'element' transform provided by the parent decorator node 1.2.2.
			NodeChildren:
	NodeLocation: 1.3
	NodeType: Parallel
	NodeFunction: This is the default fallback behavior, executed when no objects are in the specified ranges. It makes the agent rotate and move downwards simultaneously.
	NodeRequirements:
	NodeChildren:
		NodeLocation: 1.3.1
		NodeType: BTActionRotate
		NodeFunction: Rotates the agent around its Y-axis at a speed of 45 degrees per second.
		NodeRequirements:
		NodeChildren:
		NodeLocation: 1.3.2
		NodeType: BTActionMoveTowards
		NodeFunction: Moves the agent slowly downwards at a speed of 0.5 units/second. This is achieved by moving towards a target object that is a child of the agent, positioned at (0, -1, 0) locally. The 'runTillTargetIsReached' field must be 'false' to ensure this action doesn't block the tree from re-evaluating conditions on the next frame.
		NodeRequirements:
		NodeChildren:


# Tree plan for it
prompt:
a behaviour that makes the world object move towards the closest object in range of 5 to 8, but it it moves into the range of 2, move away from it, and if nothing is in any range rotate and move down slowly
Successfully generated tree plan.
Selector, Sequence, BTConditionalInRange, BTForArrayElement, BTActionMoveAway, Sequence, BTConditionalInRange, BTForArrayElement, BTActionMoveTowards, Parallel, BTActionRotate, BTActionMoveTowards
_______________________
NodeLocation: 1
NodeType: Selector
NodeFunction: Root node that selects one of three behaviors based on object proximity. It prioritizes moving away from very close objects, then moving towards objects in an optimal range, and finally defaults to a slow, rotating descent if no objects are detected.
NodeRequirements: none
NodeChildren:
	NodeLocation: 1.1
	NodeType: Sequence
	NodeFunction: This sequence handles the 'danger' scenario. It first checks for objects that are too close (within 2 units) and then moves away from the closest one.
	NodeRequirements: none
	NodeChildren:
		NodeLocation: 1.1.1
		NodeType: BTConditionalInRange
		NodeFunction: Checks for any objects within a 2-unit radius (min 0, max 2). If an object is found, this node succeeds and provides a list of all objects in range, sorted by distance.
		NodeRequirements: none
		NodeLocation: 1.1.2
		NodeType: BTForArrayElement
		NodeFunction: Decorator that extracts the first element (index 0) from the list of transforms provided by node 1.1.1. Since the list is sorted, this gives us the closest object.
		NodeRequirements: Requires the 'allObjectsInRange' list from node 1.1.1.
		NodeChildren:
			NodeLocation: 1.1.2.1
			NodeType: BTActionMoveAway
			NodeFunction: Moves the AI away from the target object provided by the parent decorator (node 1.1.2) at a speed of 4 units/sec. The 'runTillTargetIsReached' field is set to false to make the movement continuous and reactive rather than a one-time action.
			NodeRequirements: Requires the 'element' transform provided by node 1.1.2.
	NodeLocation: 1.2
	NodeType: Sequence
	NodeFunction: This sequence handles the primary objective. It checks for objects in the engagement range (5 to 8 units) and moves towards the closest one.
	NodeRequirements: none
	NodeChildren:
		NodeLocation: 1.2.1
		NodeType: BTConditionalInRange
		NodeFunction: Checks for any objects within a 5 to 8 unit radius. If an object is found, this node succeeds and provides a list of all objects in that range, sorted by distance.
		NodeRequirements: none
		NodeLocation: 1.2.2
		NodeType: BTForArrayElement
		NodeFunction: Decorator that extracts the first element (index 0) from the list of transforms provided by node 1.2.1, which corresponds to the closest object in the engagement range.
		NodeRequirements: Requires the 'allObjectsInRange' list from node 1.2.1.
		NodeChildren:
			NodeLocation: 1.2.2.1
			NodeType: BTActionMoveTowards
			NodeFunction: Moves the AI towards the target object provided by the parent decorator (node 1.2.2) at a speed of 3 units/sec. The 'runTillTargetIsReached' field is set to false to ensure the AI continuously adjusts its path towards the target.
			NodeRequirements: Requires the 'element' transform provided by node 1.2.2.
	NodeLocation: 1.3
	NodeType: Parallel
	NodeFunction: This is the default fallback behavior, executed when no other conditions are met. It runs two actions simultaneously: rotating in place and moving slowly downwards.
	NodeRequirements: none
	NodeChildren:
		NodeLocation: 1.3.1
		NodeType: BTActionRotate
		NodeFunction: Rotates the AI around its Y-axis at a constant speed of 45 degrees per second.
		NodeRequirements: none
		NodeLocation: 1.3.2
		NodeType: BTActionMoveTowards
		NodeFunction: Moves the AI downwards at a slow speed of 0.5 units/sec. This is achieved by targeting a child object positioned at (0, -1, 0) relative to the AI. The 'runTillTargetIsReached' field is set to false, which is critical for creating continuous movement, as the target moves with the AI and can never be 'reached'.
		NodeRequirements: none
5:44
.
you can see in this part:
NodeLocation: 1.2.2.1
			NodeType: BTActionMoveTowards
			NodeFunction: Moves the AI towards the target object provided by the parent decorator (node 1.2.2) at a speed of 3 units/sec. The 'runTillTargetIsReached' field is set to false to ensure the AI continuously adjusts its path towards the target.
			NodeRequirements: Requires the 'element' transform provided by node 1.2.2.
i include info about an important variable, AI decited to do so, in order to make if work for sure (edited) 
5:45
and i am using gemini 2.5 (pro and flash, depending on the part of generation, tested out a lot of models, opus works well too, groq is too weak, chat models are too stupid for advanced trees)