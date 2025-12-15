FROM maven:3.9-eclipse-temurin-17 AS build

WORKDIR /app

# Set Maven environment variables for increased timeouts and retries
ENV MAVEN_OPTS="-Dmaven.wagon.http.connectionTimeout=600000 -Dmaven.wagon.http.readTimeout=600000 -Dmaven.wagon.http.retryHandler.count=5 -Dmaven.wagon.http.pool.timeout=600000"

# Copy pom.xml first for better caching
COPY pom.xml .

# Download dependencies with retries
RUN set -e; \
    for i in 1 2 3 4 5; do \
        if mvn dependency:go-offline -B; then \
            echo "Dependencies downloaded successfully on attempt $i"; \
            break; \
        else \
            echo "Dependency download attempt $i failed, retrying..." && sleep 10; \
            if [ $i -eq 5 ]; then \
                echo "Failed to download dependencies after 5 attempts"; \
                exit 1; \
            fi; \
        fi; \
    done

# Copy source code
COPY src ./src

# Build the application with retries
RUN set -e; \
    for i in 1 2 3 4 5; do \
        if mvn clean package -DskipTests; then \
            echo "Build succeeded on attempt $i"; \
            break; \
        else \
            echo "Build attempt $i failed, retrying..." && sleep 10; \
            if [ $i -eq 5 ]; then \
                echo "Build failed after 5 attempts"; \
                exit 1; \
            fi; \
        fi; \
    done && \
    ls -la /app/target/*.jar || (echo "JAR file not found after build" && exit 1)

# Runtime stage
FROM eclipse-temurin:17-jre-alpine

WORKDIR /app

# Copy the built JAR
COPY --from=build /app/target/*.jar app.jar

# Expose port
EXPOSE 8087

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8087/actuator/health || exit 1

# Run the application
ENTRYPOINT ["java", "-jar", "app.jar"]

