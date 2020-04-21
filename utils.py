import random

# Read specific XML file
def read_xml_file(file_path):
    with open(file_path, 'r') as f:
        xml = f.read()
    return xml

# Generate random XML Mission specifications for Malmo
def generate_xml(mission_time):
    trees = list()
    for i in range(3):
        newtree = (random.randint(2,8), random.randint(4, 8))
        while newtree in trees:
            newtree = (random.randint(2,8), random.randint(4, 8))
        trees.append(newtree)

    milli_seconds_time = mission_time*60*1000

    mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

        <About>
            <Summary>Tree Cutting Mission</Summary>
        </About>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>1000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
                <AllowSpawning>false</AllowSpawning>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,2*3,2;4;" />
                <DrawingDecorator>
                    <DrawCuboid x1="0" y1="46" z1="0" x2="10" y2="50" z2="20" type="air" />            <!-- limits of our arena -->
                    <DrawCuboid x1="0" y1="45" z1="0" x2="10" y2="45" z2="20" type="grass" />           <!-- dirt floor -->
                    
                    <DrawCuboid x1="0" y1="46" z1="0" x2="0" y2="48" z2="20" type="stone" />
                    <DrawCuboid x1="10" y1="46" z1="0" x2="10" y2="48" z2="20" type="stone" />
                    <DrawCuboid x1="0" y1="46" z1="20" x2="10" y2="48" z2="20" type="stone" />
                    <DrawCuboid x1="0" y1="46" z1="0" x2="10" y2="48" z2="0" type="stone" />

                    <DrawBlock   x="4"   y="45"  z="1"  type="cobblestone" />                           <!-- the starting marker -->
                    
                    <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="45"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- the destination marker -->
                    <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="46"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- another destination marker -->
                    <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="47"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- the destination marker -->
                    <DrawBlock   x="''' +str(trees[0][0]) +'''"   y="48"  z="''' +str(trees[0][1]) +'''" type="log" />                           <!-- another destination marker -->

                    <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="45"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- the destination marker -->
                    <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="46"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- another destination marker -->
                    <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="47"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- the destination marker -->
                    <DrawBlock   x="''' +str(trees[1][0]) +'''"   y="48"  z="''' +str(trees[1][1]) +'''" type="log" />                           <!-- another destination marker -->

                    <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="45"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- the destination marker -->
                    <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="46"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- another destination marker -->
                    <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="47"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- the destination marker -->
                    <DrawBlock   x="''' +str(trees[2][0]) +'''"   y="48"  z="''' +str(trees[2][1]) +'''" type="log" />                           <!-- another destination marker -->
                </DrawingDecorator>
                <ServerQuitFromTimeUp timeLimitMs="'''+str(milli_seconds_time)+'''" />
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>Barbie</Name>
            <AgentStart>
                <Placement x="4.5" y="46.0" z="1.5" pitch="0" yaw="0"/>
                <Inventory>
                    <InventoryItem slot="0" type="diamond_axe"/>
                </Inventory>
                <!-- <Placement x="1" y="45" z="0" pitch="0" yaw="0" /> -->
            </AgentStart>
            <AgentHandlers>
                <ObservationFromFullStats />
                <DiscreteMovementCommands />
                <VideoProducer want_depth="false">
                    <Width>640</Width>
                    <Height>480</Height>
                </VideoProducer>
                <RewardForTouchingBlockType>
                    <Block reward="40.0" type="log" behaviour="constant" />
                    <Block reward="50.0" type="leaves" behaviour="constant" />
                    <Block reward="-30.0" type="stone" behaviour="constant" />
                </RewardForTouchingBlockType>
                <RewardForCollectingItem>
                    <Item reward="100" type="log"/>
                </RewardForCollectingItem>
                <AgentQuitFromTimeUp timeLimitMs="'''+str(milli_seconds_time)+'''" />
            </AgentHandlers>
        </AgentSection>
    </Mission>
    '''
    return mission_xml
