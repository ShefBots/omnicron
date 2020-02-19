import classes.env.*;
import classes.math.*;
import classes.env.nonpolyshapes.*;
import classes.graphics.SimpleDisplay;
import java.awt.Color;
import java.awt.Graphics2D;


public class BallReflection {
	public static int STEPS = 10;
	public static double rayLen = 1000;
	public static void main(String[] args) throws InterruptedException
	{
		SimpleDisplay d = new SimpleDisplay(550,600,true, true);
		Graphics2D g = d.getGraphics2D();

		double radius = 60;
		double height = 300;
		NVector offsetV = new NVector(new double[]{200,100});
		NVector cameraV = offsetV.add(new NVector(new double[]{0,height}));

		Point offset = toPoint(offsetV);
		Point camera = toPoint(cameraV);
		
		Environment env = new Environment();
		env.entities.add(new CircleBoundedEntity(offset.getX(), offset.getY(), radius));
		CircleBoundedEntity camCircle = new CircleBoundedEntity(camera.getX(), camera.getY(), 4);
		env.entities.add(camCircle);
		
		g.setColor(Color.BLACK);
		env.draw(g);
		d.repaint();
		g.setColor(Color.RED);


		////// The maths part.
		//NVector outputDirection = new NVector(new double[]{1,1}).normalize(); // Points directly horizontally to the right.

		//double a = outputDirection.dot(outputDirection);
		//double b = -2*cameraV.dot(outputDirection);
		//double c = cameraV.dot(cameraV) - radius*radius;

		//double distance = (-b + Math.sqrt(b*b - 4*a*c))/(2*a);
		//double d2 = (-b - Math.sqrt(b*b - 4*a*c))/(2*a);
		//if(d2 < distance)
		//	distance = d2;// Get the closest one.

		//// Derive the angle:
		//double angle = Math.acos((radius*radius - height*height - distance*distance)/(-2*height*distance));
		//double arbDist = 1000;
		//Point endPoint = new Point(camera.getX() + Math.sin(angle)*arbDist, camera.getY() + Math.cos(angle)*arbDist);
		////Point endPoint = new Point(100, 100);
		//DistancedHit out = env.hitScan(camera, endPoint);
		//g.drawLine((int)camera.getX(), (int)camera.getY(), (int)endPoint.getX(), (int)endPoint.getY());
		//d.repaint();


		// SCREW THE MATHS PART. WE'RE GOING 'PROXIMA ON THIS.
		double targetAngle = 
		NVector	targetSlope = ;
		double maximumAngle = Math.asin(radius/height);

		for(double a = 0.0; a<=maximumAngle; a+=(maximumAngle/STEPS))
		{
			NVector ray = new NVector(new double[]{Math.sin(a), -Math.cos(a)});
			Point endPoint = toPoint(ray.scale(rayLen).add(cameraV));
			Hit out = env.hitScan(camera, endPoint, (AbstractEntity)camCircle);

			NVector normal = toNVector(out).subtract(offsetV);
			NVector invertedRay = ray.scale(-100);
			NVector mirror = normal.scale(2 * ((invertedRay.dot(normal))/(normal.dot(normal)))).subtract(invertedRay);
			double difference = 

			// The drawing
			
			d.fill(Color.WHITE);
			g.setColor(Color.BLACK);
			env.draw(g);
			g.setColor(Color.RED);
			g.drawLine((int)camera.getX(), (int)camera.getY(), (int)out.getX(), (int)out.getY());
			g.setColor(Color.BLUE);
			g.drawLine((int)out.getX(), (int)out.getY(), (int)(out.getX()+normal.getElement(0)), (int)(out.getY()+normal.getElement(1)));
			g.setColor(Color.GREEN);
			g.drawLine((int)out.getX(), (int)out.getY(), (int)(out.getX()+mirror.getElement(0)), (int)(out.getY()+mirror.getElement(1)));
			d.repaint();
			Thread.sleep(1000);
		}
	}
	public static Point toPoint(NVector v)
	{
		return new Point(v.getElement(0), v.getElement(1));
	}
	public static NVector toNVector(StaticPoint p)
	{
		return new NVector(new double[]{p.getX(), p.getY()});
	}
}
