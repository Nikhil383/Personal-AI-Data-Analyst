from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from app.config import DATABASE_URL

Base = declarative_base()

class Customer(Base):
    __tablename__ = 'customers'

    customer_id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(255), nullable=False)
    country = Column(String(100))
    credit_limit = Column(Float)
    risk_score = Column(Float)

    invoices = relationship("Invoice", back_populates="customer")

class Invoice(Base):
    __tablename__ = 'invoices'

    invoice_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey('customers.customer_id'), nullable=False)
    invoice_date = Column(Date, nullable=False)
    due_date = Column(Date)
    amount = Column(Float, nullable=False)
    status = Column(String(50))  # paid, unpaid, overdue

    customer = relationship("Customer", back_populates="invoices")
    payments = relationship("Payment", back_populates="invoice")

class Payment(Base):
    __tablename__ = 'payments'

    payment_id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey('invoices.invoice_id'), nullable=False)
    payment_date = Column(Date, nullable=False)
    payment_amount = Column(Float, nullable=False)

    invoice = relationship("Invoice", back_populates="payments")

# Database session setup helper
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
