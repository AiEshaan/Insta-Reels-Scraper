import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from main import run_agent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ScrapingScheduler:
    def __init__(self, interval_hours=6):
        self.interval_hours = interval_hours
        self.scheduler = BackgroundScheduler()
        self.job = None
        self.username = None
        self.password = None

    def scrape_task(self):
        try:
            logging.info("Running scheduled scraping...")
            run_agent(username=self.username, password=self.password, headless=True)
            logging.info("Scheduled scraping completed successfully")
        except Exception as e:
            logging.error(f"Error in scheduled scraping: {str(e)}")

    def start(self, username, password):
        if self.job and self.job.next_run_time:
            logging.info("Scheduler already running")
            return

        self.username = username
        self.password = password

        # Remove existing job if any
        if self.job:
            self.scheduler.remove_job(self.job.id)

        # Add new job
        self.job = self.scheduler.add_job(
            self.scrape_task,
            trigger=IntervalTrigger(hours=self.interval_hours),
            next_run_time=None,  # Start after first interval
            id='scraping_job'
        )

        # Start the scheduler if it's not running
        if not self.scheduler.running:
            self.scheduler.start()

        logging.info(f"Started scraping scheduler (interval: {self.interval_hours} hours)")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logging.info("Stopped scraping scheduler")

    @property
    def is_running(self):
        return self.scheduler.running and bool(self.job and self.job.next_run_time)

# Create global scheduler instance
scheduler = ScrapingScheduler()