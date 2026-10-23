"""The fixed text of the experiment: the 50 injected concepts, the baseline words their vectors are measured against,
each concept's word forms, and the neutral passages the trials are built from."""
import re

# 50 concepts from Lindsey 2026 / Macar et al. 2026
CONCEPTS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Happiness",
    "Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
    "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
    "Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
    "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
    "Denim", "Monoliths", "Milk", "Bread", "Silver",
]

# 100 baseline words from Macar et al.'s code (duplicate "Butterflies" kept for fidelity)
BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]


# Word forms that count as naming a concept: inflections, derived forms and exact synonyms (airship = dirigible),
# not related concepts (jeans for Denim, lava for Volcanoes). Matched at a word start, case-insensitive.
CONCEPT_FORMS = {
    "Dust": ["dust"], "Satellites": ["satellite"], "Trumpets": ["trumpet"], "Origami": ["origami"],
    "Illusions": ["illusion"], "Cameras": ["camera"], "Lightning": ["lightning"], "Constellations": ["constellation"],
    "Treasures": ["treasure"], "Phones": ["phone", "telephone", "smartphone"], "Trees": ["tree"],
    "Avalanches": ["avalanche"], "Mirrors": ["mirror"], "Fountains": ["fountain"], "Quarries": ["quarry", "quarries"],
    "Sadness": ["sadness", "sad\\b"], "Xylophones": ["xylophone"], "Secrecy": ["secrecy", "secret"],
    "Oceans": ["ocean"], "Happiness": ["happiness", "happy"], "Deserts": ["desert"], "Kaleidoscopes": ["kaleidoscop"],
    "Sugar": ["sugar"], "Vegetables": ["vegetable", "veggie"], "Poetry": ["poetry", "poem", "poet"],
    "Aquariums": ["aquari"], "Bags": ["bag\\b", "bags\\b"], "Peace": ["peace"], "Caverns": ["cavern"],
    "Memories": ["memor"], "Frosts": ["frost"], "Volcanoes": ["volcan"], "Boulders": ["boulder"],
    "Harmonies": ["harmon"], "Masquerades": ["masquerade"], "Rubber": ["rubber"], "Plastic": ["plastic"],
    "Blood": ["blood"], "Amphitheaters": ["amphitheat"], "Contraptions": ["contraption"], "Youths": ["youth"],
    "Dynasties": ["dynast"], "Snow": ["snow"], "Dirigibles": ["dirigible", "airship", "zeppelin", "blimp"],
    "Algorithms": ["algorithm"], "Denim": ["denim"], "Monoliths": ["monolith"], "Milk": ["milk"], "Bread": ["bread"],
    "Silver": ["silver"],
}

# Concepts without a plural in the list are mass nouns; the rest are plurals of a count noun.
MASS = {"Dust", "Origami", "Lightning", "Sadness", "Secrecy", "Happiness", "Sugar", "Poetry", "Peace", "Rubber",
        "Plastic", "Blood", "Snow", "Denim", "Milk", "Bread", "Silver"}


def singular(c):
    w = c.lower()
    if c in MASS:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    return w[:-1]


def surface_forms(c):
    """The spellings whose probabilities add up to P(concept word): singular and as listed, lower-case and capitalised."""
    forms = {singular(c), c.lower()}
    return sorted({f for w in forms for f in (w, w.capitalize())})


_FORMS_RX = {c: re.compile(r"\b(?:" + "|".join(f) + ")", re.I) for c, f in CONCEPT_FORMS.items()}


def names_concept(text, concept):
    """The text names the concept (any form in CONCEPT_FORMS, at a word start)."""
    return bool(_FORMS_RX[concept].search(text))


# ---------------------------------------------------------------- passages
# Everyday, low-imagery prose on unrelated topics; each trial shows one passage twice. None names any of the 50
# concepts (checked before every run). CALIBRATION passages are used only to set each concept's strength; TEST
# passages only for the reported curves.
CALIBRATION = [
    # bus timetable
    "Starting next Monday, route 14 will leave the central station every twenty minutes instead of every fifteen. "
    "The first departure moves to 6:10 in the morning, and the last one to 11:40 at night. Weekend service is "
    "unchanged. Passengers who transfer to route 9 at Elm Street should allow a few extra minutes, because the two "
    "schedules no longer line up exactly.",
    # printer instructions
    "To print from your laptop, first connect to the office network and open the settings panel. Choose the device "
    "labelled Floor 3 East and set it as the default. If a job gets stuck in the queue, cancel it from the panel "
    "rather than switching the machine off. Double-sided printing is on by default; you can turn it off for single "
    "pages.",
    # neighbourhood meeting minutes
    "The residents' association met on Thursday evening with eleven members present. The finance officer reported that "
    "the annual fees had mostly been collected. A proposal to repaint the benches along the main path was approved, "
    "with work to begin in spring. The next meeting will be held on the second Thursday of the month at the "
    "community hall.",
    # chess opening
    "In this opening, White advances the king's pawn two squares and Black answers symmetrically. White then "
    "develops the knight toward the center, attacking the pawn, and Black defends by bringing out a knight of their "
    "own. Both sides aim to castle early, keep their pieces coordinated, and avoid moving the same piece twice "
    "before development is complete.",
    # sleep advice
    "Most adults do best with seven to nine hours of sleep a night. Going to bed and waking up at the same time "
    "each day, even on weekends, helps the body keep a steady rhythm. Screens late in the evening can make it "
    "harder to fall asleep, so it helps to put devices away half an hour before bed and keep the bedroom cool "
    "and quiet.",
    # software release notes
    "Version 4.2 fixes a problem that caused the app to close unexpectedly when exporting large reports. Search "
    "results now load faster, and the settings page has been reorganized so that account options appear first. "
    "Users on older operating systems will see a notice asking them to update, since support for those versions "
    "ends later this year.",
    # insurance claim
    "To file a claim, fill out the online form within thirty days of the incident and attach any receipts or "
    "photographs you have. An adjuster will contact you within five business days to review the details. Keep a "
    "copy of everything you submit. Payments are usually issued within two weeks once the claim has been "
    "approved.",
    # language learning
    "Learning a new language takes steady practice more than long sessions. Ten or fifteen minutes each day of "
    "listening and speaking tends to work better than a few hours once a week. Many learners find it useful to "
    "label objects around the house, repeat short phrases out loud, and review vocabulary right before going "
    "to bed.",
]

TEST = [
    # tax deadline
    "The deadline for filing individual returns is the fifteenth of April. If you need more time, you can request "
    "an extension, but any amount owed is still due by the original date. Interest is charged on late payments. "
    "Keep your records for at least three years in case the office has questions about a return you have "
    "already filed.",
    # library hours
    "The public library is open from nine to eight on weekdays and from ten to five on Saturdays. It is closed on "
    "Sundays and public holidays. Members may borrow up to twelve items at a time for three weeks, and most items "
    "can be renewed online twice. Study rooms can be booked up to seven days in advance at the front desk.",
    # bookshelf assembly
    "Lay out all the panels and check them against the parts list before you begin. Attach the two side panels to "
    "the base with the long screws, then slide the back panel into the grooves. Fit the shelves at the heights you "
    "want, using the small pegs provided. Tighten every screw only once the whole frame is square.",
    # football match
    "The home side took the lead after twenty minutes with a header from a corner. The visitors pressed hard after "
    "the break and equalized midway through the second half. Both teams had chances late on, but neither could "
    "find a winner, and the match ended level. The result keeps both clubs in the middle of the table.",
    # lentil soup
    "Rinse a cup of lentils and set them aside. In a large pot, soften a chopped onion and two carrots in a little "
    "oil, then add garlic and a spoonful of cumin. Stir in the lentils with four cups of water and simmer for "
    "about half an hour, until tender. Season with salt, pepper and a squeeze of lemon before serving.",
    # office move
    "Our team will move to the fourth floor at the end of the month. Please pack your desk into the boxes provided "
    "and label each one with your name and new desk number. Computers and monitors will be moved by the technical "
    "staff over the weekend. Everything should be set up and working by Monday morning.",
    # train delay notice
    "Due to maintenance work on the line, trains between the two northern stations will run on a reduced "
    "schedule this weekend. A replacement coach service will operate every half hour. Journeys may take up to "
    "forty minutes longer than usual, so passengers are advised to check times before they travel and to allow "
    "extra time.",
    # houseplant repotting
    "Repot a houseplant when its roots begin to circle the bottom of the container. Choose a new pot only slightly "
    "larger than the old one, with a drainage hole. Loosen the root ball gently, set the plant at the same depth "
    "as before, and fill in around it with fresh potting mix. Water it well and keep it out of direct sun for "
    "a week.",
    # meeting scheduling email
    "Thanks for getting back to me. Tuesday afternoon works well for our side, so let's plan on two o'clock. I "
    "will send a calendar invitation with the call details shortly. If anything changes, just let me know and we "
    "can find another time later in the week. I'll circulate a short agenda the day before.",
    # gym membership terms
    "Membership renews automatically each month unless you cancel at least ten days before the renewal date. You "
    "can pause your membership for up to two months a year for travel or illness. Guests are welcome on weekdays "
    "for a small fee. Lockers are available on a first-come basis and must be emptied at the end of each day.",
    # car maintenance
    "Check the tire pressure once a month and before long trips, using the figures printed inside the driver's "
    "door. Change the engine oil roughly every five thousand miles, or as the manual recommends. Replace the wiper "
    "blades once a year, and have the brakes inspected whenever you hear grinding or the pedal feels soft.",
    # conference registration
    "Registration for the annual conference opens on the first of the month. Early registration costs less and "
    "closes six weeks before the event. Speakers should upload their slides by the Friday before the conference. "
    "Lunch is provided on both days, and a list of nearby hotels with discounted rates is available on the "
    "event website.",
]

# More test passages of the same kind, used when a sweep is run with --passages above 12 (the 12 in TEST come first,
# so results from the original set keep their passage numbers).
TEST_EXTRA = [
    # parking permit
    "Residential parking permits expire at the end of June and must be renewed each year. You can renew online or "
    "at the city office with proof of address and your vehicle registration. Each household may hold up to two "
    "permits. Cars without a valid permit may be ticketed after the first week of July.",
    # password reset
    "If you forget your password, choose the reset link on the sign-in page and enter the email address on your "
    "account. A message with a six-digit code will arrive within a few minutes. Enter the code, then pick a new "
    "password of at least twelve characters. The code stops working after fifteen minutes.",
    # recycling schedule
    "Recycling is collected every other Wednesday, on the same day as regular waste. Paper and cardboard go in the "
    "blue bin, and glass and metal cans go in the green one. Bins should be at the curb by seven in the morning. "
    "When a holiday falls on a weekday, collection moves one day later that week.",
    # thermostat
    "To set a weekly schedule on the thermostat, press the menu button and choose the program option. Pick the days "
    "you want, then enter a wake time, a leave time, a return time and a sleep time, each with its own temperature. "
    "Press save when you are done. You can override the schedule at any point by using the arrow keys.",
    # expense reports
    "Expense reports are due by the fifth working day of each month. Attach a receipt for every item over twenty "
    "dollars and note the purpose of the trip or purchase. Meals are reimbursed up to the daily limit set in the "
    "travel policy. Approved reports are paid with the next payroll run.",
    # dishwasher
    "Scrape plates before loading them, but there is no need to rinse them first. Put cups and glasses on the top "
    "rack and plates, pans and cutlery on the bottom. Make sure nothing blocks the spray arm from turning. Clean the "
    "filter at the base of the machine about once a month.",
    # voter registration
    "To vote in the coming election you must be registered at your current address at least thirty days beforehand. "
    "You can check your status on the county website using your name and date of birth. If you have moved, submit "
    "an updated form online or by mail. Polling places are open from seven in the morning until eight at night.",
    # warranty
    "This product is covered against defects in materials and workmanship for two years from the date of purchase. "
    "The warranty does not cover damage caused by misuse, accidents or unauthorized repairs. To make a claim, "
    "contact customer support with your order number. We will repair or replace the item at no cost to you.",
    # lease renewal
    "Your lease ends on the last day of August. If you would like to stay, please sign and return the renewal form "
    "by the first of July. The monthly rent for the new term will rise by three percent. Tenants who plan to leave "
    "must give sixty days of written notice and arrange a final inspection.",
    # job application
    "Applications for the analyst position are accepted through the careers page until the end of the month. Please "
    "upload a resume and a one-page cover letter in a single file. Shortlisted candidates will be contacted within "
    "two weeks to arrange a first interview. We are unable to reply to every applicant individually.",
    # clinic appointment
    "This is a reminder of your appointment on Thursday at ten fifteen in the morning. Please arrive ten minutes "
    "early to complete the check-in form, and bring your insurance card and a list of current medications. If you "
    "cannot attend, let us know at least one day ahead so the slot can be offered to someone else.",
    # luggage allowance
    "Each passenger may bring one carry-on item and one small personal item on board at no charge. The carry-on "
    "must fit in the overhead compartment and weigh no more than ten kilograms. Checked luggage can be added during "
    "booking or at the airport for a higher fee. Liquids must be in containers of one hundred milliliters or less.",
    # electricity bill
    "Your bill has two parts: a fixed daily charge and a charge for the energy you used. Usage is measured in "
    "kilowatt-hours and read from the meter every two months. If the meter could not be read, the amount is "
    "estimated from past use and corrected on the next bill. Payment is due twenty-one days after the issue date.",
    # swimming pool rules
    "The pool is open to members from six in the morning until nine at night. Please shower before entering the "
    "water, and tie back long hair. Children under twelve must be accompanied by an adult at all times. The lanes on "
    "the left are reserved for lap swimming during the early morning session.",
    # course enrollment
    "Enrollment for the spring term opens on the second Monday of November. Students register in order of the "
    "credits they have completed, starting with final-year students. Each course lists its meeting times, its "
    "prerequisites and the number of places left. You may change your selection without penalty until the end of "
    "the first week of classes.",
    # jury duty
    "You have been selected for jury service beginning on the date shown above. Please report to the courthouse by "
    "eight thirty and bring this notice with you. Service usually lasts one day unless you are assigned to a trial. "
    "If the date causes serious hardship, you may request one postponement through the court's website.",
    # hotel policy
    "Check-in begins at three in the afternoon and check-out is by eleven in the morning. A late check-out can be "
    "arranged at the front desk when rooms are available. Breakfast is served in the dining room from seven until "
    "ten. Reservations can be cancelled without charge up to two days before arrival.",
    # software license
    "This license allows you to install the software on up to three devices that you own. You may not copy it for "
    "others, rent it out or attempt to reverse engineer it. Updates released during the license period are included "
    "at no extra cost. The license ends automatically if any of these terms are broken.",
    # water main repair
    "Crews will repair a water main on Maple Avenue between nine in the morning and four in the afternoon on "
    "Tuesday. Water service to nearby homes will be shut off during that time. Residents may wish to fill a few "
    "containers beforehand. When service returns, run the cold tap for a minute or two until the water is clear.",
    # pet licensing
    "All dogs over four months old must be licensed with the city. To apply, provide proof of a current rabies "
    "vaccination and pay the annual fee, which is lower for animals that have been neutered. The tag should be "
    "attached to the collar at all times. Licenses are renewed each January.",
    # bank fees
    "The basic checking account has no monthly fee when the balance stays above five hundred dollars. Otherwise a "
    "charge of six dollars applies at the end of the statement period. Withdrawals at our own machines are free, "
    "and other banks may add their own charge. Paper statements cost two dollars a month, while electronic ones "
    "are free.",
    # passport renewal
    "You can renew a passport by mail if your current one was issued within the last fifteen years and is not "
    "damaged. Send the completed form, your old passport, one recent photograph and the fee. Routine processing "
    "takes six to eight weeks. A faster service is available for an added charge.",
    # road resurfacing
    "Resurfacing work on Oak Street will begin next Monday and is expected to last two weeks. One lane will stay "
    "open in each direction, though delays are likely at busy times. Street parking will not be allowed between "
    "seven in the morning and five in the afternoon. Drivers are encouraged to use other routes where possible.",
    # volunteer shifts
    "Volunteers are needed for three shifts on the day of the event: morning setup, afternoon registration and "
    "evening cleanup. Each shift lasts about three hours. You can sign up for one or more through the form linked "
    "below. A short briefing will be held half an hour before each shift begins.",
    # school pickup
    "Dismissal is at three fifteen. Parents collecting children by car should enter from the north gate and follow "
    "the signs to the pickup lane. Please stay in your vehicle and display the family name card on the dashboard. "
    "Children who have not been collected by three forty-five will wait in the main office.",
    # laundry room
    "The laundry room is open from seven in the morning to ten at night. Machines take payment by card only. Please "
    "remove your clothes promptly when a cycle ends so that others can use the machine. Clean the lint screen after "
    "each use of a dryer, and report any faults to the building manager.",
    # store returns
    "Items may be returned within thirty days of purchase with the original receipt. Refunds are made to the "
    "original form of payment. Without a receipt, we can offer an exchange or store credit at the current price. "
    "Clearance items and gift cards cannot be returned.",
    # fire drill
    "A fire drill will take place on Wednesday morning. When the alarm sounds, stop work, leave by the nearest "
    "stairway and do not use the elevators. Gather at the assembly point in the east parking lot and wait for your "
    "floor warden to take attendance. Do not go back inside until the all-clear is given.",
    # budget meeting
    "The committee reviewed the draft budget for the coming year. Spending on maintenance is set to rise by four "
    "percent, mainly because of roof repairs. Income from fees is expected to stay level. Members agreed to "
    "postpone the purchase of new furniture and to vote on the final budget at the next meeting.",
    # wireless network setup
    "Plug the router into the wall socket and connect it to the modem with the cable provided. Wait until the "
    "status light turns solid green. On your computer, choose the network name printed on the label and enter the "
    "password shown beneath it. You can change both later from the settings page.",
    # bicycle lane rules
    "Cyclists should ride in the marked lane where one is provided and travel in the same direction as traffic. Use "
    "a front light and a rear light after dark. Signal with your arm before turning or changing lanes. Riding on "
    "the sidewalk is not permitted in the downtown area.",
    # vaccination clinic
    "A flu vaccination clinic will be held in the community center on Saturday from nine until one. No appointment "
    "is needed, and the vaccine is free for residents. Please wear a short-sleeved shirt and bring identification. "
    "You will be asked to wait for fifteen minutes afterwards before leaving.",
    # elevator maintenance
    "The north elevator will be out of service on Thursday for its annual inspection. The south elevator will keep "
    "running throughout, though waits may be longer than usual. Deliveries should be scheduled for another day "
    "where possible. Work is expected to finish by five in the afternoon.",
    # newsletter subscription
    "You are receiving this message because you signed up for our monthly newsletter. Each issue includes product "
    "updates, upcoming events and a short how-to guide. To change how often you hear from us, visit the preferences "
    "page. You can unsubscribe at any time using the link at the bottom of every email.",
    # rental car
    "The rental period starts at the pickup time shown on your agreement and is charged in blocks of one day. "
    "Return the car with a full tank to avoid a refueling fee. Additional drivers must be listed on the agreement "
    "and show a valid license. Report any accident to the rental office as soon as possible.",
    # time sheets
    "Time sheets must be submitted by noon on the last Friday of each pay period. Record your start time, your end "
    "time and any unpaid breaks for every day worked. Overtime needs approval from your supervisor in advance. "
    "Late submissions may be held over until the following pay date.",
    # basketball game
    "The visitors led by eight points at halftime after a strong second quarter. The home team answered with a "
    "run of twelve straight points early in the third and never trailed again. Their center finished with twenty "
    "points and eleven rebounds. The win moves them into third place in the division.",
    # filter coffee
    "Bring fresh water to a boil and let it rest for half a minute. Place a paper filter in the cone and rinse it "
    "with a little hot water. Add two spoonfuls of ground coffee for each cup, then pour the water slowly in "
    "circles until the cup is full. The whole process should take about three minutes.",
]
