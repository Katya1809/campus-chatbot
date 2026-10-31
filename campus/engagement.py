"""Campus Engagement Opportunities & Leadership Guide for IIT Bhubaneswar.

Covers:
  - 4-Year Student Engagement Journey (Freshman to Senior year progression)
  - General Championship (GC) Strategy & Scoring Guide
  - Extra Academic Activities (EAA) Navigator (NSO, NSS, NCC, Yoga)
  - Fest Organizing Committee & Leadership Portfolios (Alma Fiesta, Wissenaire, E-Summit, Pravaah)
  - Social & Community Service Pathways (Souls for Solace, UBA)
"""
from __future__ import annotations

from typing import Any


class EngagementService:
    """Service providing campus engagement roadmaps, GC guides, and leadership opportunities."""

    def __init__(self):
        pass

    def four_year_roadmap(self, year: str = "all") -> dict[str, Any]:
        """Return milestone recommendations across the 4-year undergraduate lifecycle."""
        roadmap = {
            "year1": {
                "title": "Year 1: Explorer & Learner (Foundations)",
                "focus": "Orientation, Skill Exploration, EAA Completion, and First-Year Inductions",
                "key_milestones": [
                    "Participate actively in Freshers Orientation and connect with your Department Student Guide.",
                    "Select your compulsory EAA stream (NSO Sports, NSS Social Service, or NCC) and pass the mandatory Yoga practical.",
                    "Attend open workshops and induction tryouts for 2-3 societies (1 Technical, 1 Cultural/Sports, 1 Hobby).",
                    "Compete in the General Championship (GC) Fresher events and inter-hostel leagues.",
                    "Join volunteer sub-committees for Alma Fiesta or Wissenaire to understand fest operations.",
                ],
                "recommended_time": "3-5 hours/week alongside academics (maintain CGPA 7.5+)",
                "avoid": "Over-committing to more than 3 societies; missing compulsory EAA morning attendance.",
            },
            "year2": {
                "title": "Year 2: Contributor & Core Team Member (Deepening)",
                "focus": "Technical Projects, Fest Sub-Coordinatorship, Inter-IIT Contingent Selection",
                "key_milestones": [
                    "Step up as Core Team Member in your chosen society; help design workshops or mentor freshers.",
                    "Attend selection trials for Inter-IIT Sports Meet, Inter-IIT Tech Meet, or Inter-IIT Cultural Meet.",
                    "Participate in national hackathons (HackOdisha, Smart India Hackathon) or design competitions.",
                    "Apply for Fest Sub-Coordinator roles (Logistics, Sponsorship, Events, Media & Publicity).",
                    "Explore faculty research labs, DIC projects, and prepare for 3rd-year summer internship applications.",
                ],
                "recommended_time": "6-8 hours/week",
                "avoid": "Neglecting core departmental engineering coursework.",
            },
            "year3": {
                "title": "Year 3: Leader & Fest Coordinator (Impact)",
                "focus": "Gymkhana Elections, Society Secretary Roles, Fest Heads, CDC Internships",
                "key_milestones": [
                    "Run for Gymkhana leadership: Society Secretary, General Secretary (GSec), or Hostel Rep.",
                    "Serve as Overall Coordinator or Head for Alma Fiesta, Wissenaire, E-Summit, or Pravaah.",
                    "Lead inter-departmental teams in the General Championship (GC) for your branch.",
                    "Coordinate with the Career Development Cell (CDC) for corporate internships.",
                    "Publish technical papers, patent innovations through DIC, or launch student startups via E-Cell.",
                ],
                "recommended_time": "8-10 hours/week with high leadership autonomy",
                "avoid": "Burnout during the crucial autumn placement/internship season.",
            },
            "year4": {
                "title": "Year 4: Mentor, Governor & Alumnus-in-Making (Legacy)",
                "focus": "Gymkhana Advisory, Capstone Projects, Final Placements, Alumni Connect",
                "key_milestones": [
                    "Serve as Society Governor or Senior Advisory Board member guiding elected junior secretaries.",
                    "Complete final placement drives or prepare for higher studies (GATE, GRE, CAT).",
                    "Execute the final-year B.Tech Capstone Project (BTP).",
                    "Connect with the Alumni Cell and guide freshers through formal mentorship programs.",
                    "Leave an enduring legacy: document best practices and celebrate convocation.",
                ],
                "recommended_time": "Flexible; mentor-driven commitment",
                "avoid": "Losing touch with student community initiatives.",
            },
        }

        if year in ("1", "year1", "fresher"):
            return {"year1": roadmap["year1"]}
        if year in ("2", "year2", "sophomore"):
            return {"year2": roadmap["year2"]}
        if year in ("3", "year3", "junior"):
            return {"year3": roadmap["year3"]}
        if year in ("4", "year4", "senior"):
            return {"year4": roadmap["year4"]}
        return roadmap

    def general_championship_guide(self) -> dict[str, Any]:
        """Return the complete General Championship (GC) structure and participation strategy."""
        return {
            "title": "IIT Bhubaneswar General Championship (GC) Guide",
            "overview": (
                "The General Championship (GC) is the highest inter-departmental sporting, cultural, "
                "and technological tournament. The winning department lifts the prestigious GC Rolling Trophy."
            ),
            "pillars": {
                "Technical GC": {
                    "weight": "35%",
                    "events": [
                        "Robotics Battles & Line Following (with RISC)",
                        "Algorithmic CodeSprint (with Neuromancers)",
                        "Web & Product Design Sprints (with WebND)",
                        "Case Study & FinTech Valuation (with SFEB)",
                        "Hardware Hackathon & CAD Challenges",
                    ],
                    "how_to_participate": "Nominated by Department Representatives (DepReps). Tryouts held at semester start.",
                },
                "Cultural GC": {
                    "weight": "35%",
                    "events": [
                        "Inter-Department Nukkad Natak (Street Theatre)",
                        "Stage Play & Mono Acting",
                        "Western & Eastern Group Dance",
                        "Band Performance & Solo Vocals",
                        "Fine Arts, Live Canvas & Rangoli",
                        "Parliamentary Debate & General Quiz",
                    ],
                    "how_to_participate": "Auditions conducted by departmental cultural secretaries.",
                },
                "Sports GC": {
                    "weight": "30%",
                    "events": [
                        "Cricket, Football, Basketball, Volleyball",
                        "Badminton, Table Tennis, Squash, Tennis",
                        "Athletics Track & Field (Sprints, Relays, Long Jump)",
                        "Chess & Tug of War",
                    ],
                    "how_to_participate": "Open selection matches organized by the Sports Council at the Main Sports Complex.",
                },
            },
            "scoring_system": [
                "1st Place (Gold): 100 points",
                "2nd Place (Silver): 60 points",
                "3rd Place (Bronze): 40 points",
                "Participation bonus: 10 points per contested event",
                "Spirit of Sportsmanship bonus: Up to 20 bonus points per council",
            ],
            "tips_for_freshers": [
                "Every single point scored counts toward your department's championship standing.",
                "Freshers events carry equal weight to senior rounds — an active fresher contingent often decides the GC winner!",
                "Get in touch with your branch Department Representative (DepRep) during the first two weeks of class.",
            ],
        }

    def extra_academic_activities_guide(self) -> dict[str, Any]:
        """Return comprehensive EAA details (compulsory for first-year students)."""
        return {
            "title": "Extra Academic Activities (EAA) & Compulsory Yoga Guide",
            "overview": (
                "All first-year B.Tech and ITEP students must complete one EAA stream (NSO, NSS, or NCC) "
                "along with a mandatory Yoga course. Completing EAA is an academic graduation requirement."
            ),
            "options": [
                {
                    "name": "NSO (National Sports Organisation)",
                    "activities": "Conditioning and coaching in Football, Cricket, Basketball, Volleyball, Athletics, Badminton, TT, Tennis.",
                    "timing": "Thrice weekly, 6:30 PM - 7:30 PM at the Main Sports Complex.",
                    "key_benefit": "Direct feeder for Inter-IIT sports contingent trials with professional NIS coaches.",
                },
                {
                    "name": "NSS (National Service Scheme)",
                    "activities": "Village mentorship, rural digital literacy, campus plantation, e-waste recycling, cleanliness drives.",
                    "timing": "Weekend mornings (Saturday / Sunday 9:00 AM - 11:30 AM).",
                    "key_benefit": "Grassroots social impact, leadership certification, community engagement hours.",
                },
                {
                    "name": "NCC (National Cadet Corps)",
                    "activities": "Drill practice, ceremonial guard of honour, physical fitness circuits, map reading, obstacle courses.",
                    "timing": "Twice weekly, 6:30 AM - 8:00 AM at the Parade Grounds.",
                    "key_benefit": "Discipline, leadership development, eligibility for NCC B & C certification.",
                },
            ],
            "compulsory_yoga": (
                "Held every semester for first-years. Passing the practical asana and pranayama exam is mandatory. "
                "Attendance modifies your final course score on a linear continuous scale from -5 to +5 marks."
            ),
        }

    def fest_leadership_opportunities(self) -> list[dict[str, Any]]:
        """Return recruitment and volunteering calls for major campus festivals."""
        return [
            {
                "fest": "Alma Fiesta (Annual Socio-Cultural Fest)",
                "scale": "Eastern India's premier socio-cultural festival (35,000+ footfall)",
                "recruitment_timeline": "September-October (Volunteer call) & July-August (Core team selection)",
                "sub_committees": [
                    "Events & Pro-Nites: Celebrity artist liaison, stage production, competition execution",
                    "Sponsorship & PR: Corporate pitching, brochure drafting, brand partnerships",
                    "Media & Publicity: Social media campaigns, fest teasers, poster design, campus ambassadors",
                    "Logistics & Hospitality: Guest artist accommodation, transportation, backstage management",
                    "Web & Creative Design: Portal development, registration management, 3D animations",
                ],
                "contact": "almafiesta@iitbbs.ac.in",
            },
            {
                "fest": "Wissenaire (Annual Techno-Management Fest)",
                "scale": "Flagship technology festival featuring RoboWars, Hackathons, Guest Lectures",
                "recruitment_timeline": "September-October (Junior executives) & August (Core committee)",
                "sub_committees": [
                    "Technical Competitions: Yanthrix arena fabrication, hackathon tracks, problem statements",
                    "Exhibitions & Drone GP: Defense robotics showcase, auto expos, FPV racing tracks",
                    "Workshops & Guest Lectures: Nobel Laureate / industry leader invites, tech masterclasses",
                    "Sponsorship & Marketing: Tech sponsor acquisition, merchandise distribution",
                    "Web Development & Systems: Cloud infrastructure, live leaderboards, ticketing",
                ],
                "contact": "wissenaire@iitbbs.ac.in",
            },
            {
                "fest": "E-Summit (Entrepreneurship Summit)",
                "scale": "Pan-India startup conclave bringing together venture capitalists, angel investors, founders",
                "recruitment_timeline": "October & January",
                "sub_committees": [
                    "Pitch Arena & Investor Relations: Deal flow curation, startup due diligence",
                    "Startup Internship Fair: Hiring drives, recruiter onboarding",
                    "Case Challenges: Business case writing, jury evaluation",
                ],
                "contact": "ecell@iitbbs.ac.in",
            },
            {
                "fest": "Pravaah (Inter-Collegiate Sports Fest)",
                "scale": "Premier athletic championship hosting 1,200+ collegiate athletes in 13 sports",
                "recruitment_timeline": "October-November",
                "sub_committees": [
                    "Tournament Operations: Fixture scheduling, referee liaison, scoreboards",
                    "Ground & Arena Maintenance: Turf conditioning, court lighting",
                    "Hospitality & Medical: First aid physiotherapists, hostel accommodation",
                ],
                "contact": "sports.council@iitbbs.ac.in",
            },
        ]

    def describe_engagement(self, query: str = "") -> str:
        """Synthesize campus engagement opportunities into a rich Markdown summary."""
        q = query.lower()
        if "gc" in q or "general championship" in q:
            gc = self.general_championship_guide()
            lines = [
                f"### 🏆 {gc['title']}\n",
                f"{gc['overview']}\n",
                "#### Council Pillars & Weights:",
            ]
            for p, d in gc["pillars"].items():
                lines.append(f"• **{p}** ({d['weight']}): {', '.join(d['events'][:3])} — *{d['how_to_participate']}*")
            lines.append("\n#### Scoring System:")
            for s in gc["scoring_system"]:
                lines.append(f"- {s}")
            lines.append("\n#### Tips for Freshers:")
            for t in gc["tips_for_freshers"]:
                lines.append(f"• {t}")
            return "\n".join(lines)

        if "eaa" in q or "nso" in q or "nss" in q or "ncc" in q or "yoga" in q:
            eaa = self.extra_academic_activities_guide()
            lines = [
                f"### 🏃 {eaa['title']}\n",
                f"{eaa['overview']}\n",
                "#### Available Streams:",
            ]
            for opt in eaa["options"]:
                lines.append(
                    f"• **{opt['name']}**\n"
                    f"  Activities: {opt['activities']}\n"
                    f"  Timing: {opt['timing']}\n"
                    f"  Key Benefit: {opt['key_benefit']}"
                )
            lines.append(f"\n**Compulsory Yoga Course**: {eaa['compulsory_yoga']}")
            return "\n".join(lines)

        if "fest" in q or "volunteer" in q or "alma" in q or "wissenaire" in q or "organize" in q:
            fests = self.fest_leadership_opportunities()
            lines = ["### 🎪 Campus Fest Organizing & Leadership Opportunities\n"]
            for f in fests:
                lines.append(
                    f"• **{f['fest']}**\n"
                    f"  Scale: {f['scale']}\n"
                    f"  Recruitment Timeline: {f['recruitment_timeline']}\n"
                    f"  Sub-Committees: {'; '.join(f['sub_committees'][:3])}\n"
                    f"  Contact: {f['contact']}\n"
                )
            return "\n".join(lines)

        # Default: broad overview
        r = self.four_year_roadmap()
        lines = [
            "### 🌟 IIT Bhubaneswar Campus Engagement Opportunities\n",
            "Here is how you can build an impactful 4-year campus journey:\n",
        ]
        for _, y in r.items():
            lines.append(
                f"**{y['title']}**\n"
                f"Focus: {y['focus']}\n"
                f"Key Action Steps: {y['key_milestones'][0]}; {y['key_milestones'][1]}\n"
            )
        lines.append(
            "\n💡 *Ask me about General Championship (GC) points, EAA sports/yoga options, "
            "or fest organizing teams for specific guidance!*"
        )
        return "\n".join(lines)
